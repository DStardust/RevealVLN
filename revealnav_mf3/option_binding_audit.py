"""Outcome-blind construction and auditing of MF3ZR option bindings.

The current sealed visual labels contain instruction factors but no explicit
candidate-specific binding.  This module therefore emits a review-ready
source and conservative ``UNRESOLVED`` provisional edges.  It never turns a
rank, alias, embedding, or route fact into a SUPPORTS edge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import hashlib
import json

from .option_binding_schema import (
    BindingState,
    OptionEvidenceBinding,
    OptionIdentity,
    source_commitment,
)
from .option_identity import build_option_identities, validate_binding_step


ROLE_DEC = "DEC_REQUIRED"
ROLE_PRE = "PREREQUISITE_ONLY"
ACTIVE_ROLES = frozenset({ROLE_DEC, ROLE_PRE})


class BindingAuditError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ref(root: Path, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BindingAuditError("source image reference is not a mapping")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise BindingAuditError("source image reference lacks a path")
    path = (root / path_value).resolve()
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise BindingAuditError(f"source image is not project-local: {path_value}")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or _sha256_file(path) != digest:
        raise BindingAuditError(f"source image hash mismatch: {path_value}")
    return {
        "path": str(path.relative_to(root.resolve())),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def _safe_constraint(raw: Mapping[str, object]) -> dict[str, object]:
    allowed = {"constraint_id", "kind", "subject", "relation", "object", "dependencies", "decisive_for"}
    if set(raw) != allowed:
        raise BindingAuditError("constraint graph schema drift")
    # decisive_for is retained only as an auditable source field.  It is not
    # consumed as truth because existing values are often empty/free text.
    return {
        "constraint_id": str(raw["constraint_id"]),
        "kind": str(raw["kind"]),
        "subject": str(raw["subject"]),
        "relation": None if raw["relation"] is None else str(raw["relation"]),
        "object": None if raw["object"] is None else str(raw["object"]),
        "dependencies": [str(value) for value in raw["dependencies"]],
        "decisive_for": [str(value) for value in raw["decisive_for"]],
    }


def _safe_prefix(root: Path, raw: Mapping[str, object], *, decision_step: int) -> dict[str, object]:
    step = raw.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > decision_step:
        raise BindingAuditError("prefix is not strictly causal")
    candidate_ids = raw.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids or any(not isinstance(value, str) or not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise BindingAuditError("prefix candidate aliases are invalid")
    storyboard = _safe_ref(root, raw.get("causal_storyboard"))
    panorama = _safe_ref(root, raw.get("current_panorama"))
    return {
        "step": int(step),
        "candidate_ids": list(candidate_ids),
        "causal_storyboard": storyboard,
        "current_panorama": panorama,
    }


def _prefix_identity_rows(prefixes: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for prefix in prefixes:
        safe = {
            "step": int(prefix["step"]),
            "candidate_ids": list(prefix["candidate_ids"]),
            "causal_storyboard_sha256": prefix["causal_storyboard"]["sha256"],
            "current_panorama_sha256": prefix["current_panorama"]["sha256"],
        }
        safe["source_commitment"] = source_commitment(safe)
        rows.append(safe)
    return rows


def build_review_source(
    *, root: Path, population_rows: Sequence[Mapping[str, object]], label_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], tuple[OptionIdentity, ...], tuple[OptionEvidenceBinding, ...], dict[str, object]]:
    """Build a fixed, review-ready source without assigning semantic states."""

    by_event = {str(row.get("event_id")): row for row in label_rows}
    if len(by_event) != len(label_rows):
        raise BindingAuditError("duplicate visual-label event identity")
    review_events: list[dict[str, object]] = []
    identities: list[OptionIdentity] = []
    edges: list[OptionEvidenceBinding] = []
    reason_counts: Counter[str] = Counter()
    for row in population_rows:
        event_id = str(row["event_id"])
        label = by_event.get(event_id)
        if label is None:
            raise BindingAuditError(f"missing visual label for {event_id}")
        dataset = str(row["dataset"])
        decision_step = int(row["decision_step"])
        graph = [_safe_constraint(item) for item in label.get("constraint_graph", ())]
        roles = {str(cid): str(role) for cid, role in row.get("constraint_roles", {}).items()}
        prefixes = [_safe_prefix(root, item, decision_step=decision_step) for item in label.get("prefix_sources", ())]
        if not prefixes or int(prefixes[-1]["step"]) != decision_step:
            raise BindingAuditError(f"prefix window drift for {event_id}")
        prefix_identity = _prefix_identity_rows(prefixes)
        identity_input = [
            {
                "step": item["step"],
                "candidate_ids": item["candidate_ids"],
                "source_commitment": item["source_commitment"],
            }
            for item in prefix_identity
        ]
        event_identities, identity_issues = build_option_identities(event_id, identity_input)
        identities.extend(event_identities)
        if identity_issues:
            reason_counts.update("UNSUPPORTED_OPTION_IDENTITY" for _ in identity_issues)
        option_aliases = tuple(str(value) for value in row.get("option_ids", ()))
        if tuple(prefixes[-1]["candidate_ids"]) != option_aliases:
            raise BindingAuditError(f"current candidate aliases drift for {event_id}")
        identity_by_candidate = {item.candidate_id: item for item in event_identities}
        active_constraints = tuple(cid for cid, role in roles.items() if role in ACTIVE_ROLES)
        missing = tuple(str(value) for value in row.get("independent_missing_constraints", ()))
        if missing:
            reason_counts.update("UNSUPPORTED_MISSING_DEC_BINDING" for _ in missing)
        event_edges: list[dict[str, object]] = []
        for prefix in prefixes:
            step = int(prefix["step"])
            current_candidates = tuple(str(value) for value in prefix["candidate_ids"])
            for rank, candidate_id in enumerate(current_candidates):
                identity = identity_by_candidate.get(candidate_id)
                if identity is None:
                    raise BindingAuditError("candidate identity disappeared from causal prefix")
                validate_binding_step(identity, step)
                option_id = identity.option_id
                for constraint_id in active_constraints:
                    edge_payload = {
                        "event_id": event_id,
                        "prefix_step": step,
                        "option_id": option_id,
                        "candidate_id": candidate_id,
                        "candidate_rank": rank,
                        "constraint_id": constraint_id,
                        "review_source": "PENDING_INDEPENDENT_OPTION_BINDING_REVIEW",
                    }
                    commitment = source_commitment(edge_payload)
                    edge = OptionEvidenceBinding(
                        event_id=event_id,
                        prefix_step=step,
                        option_id=option_id,
                        candidate_id=candidate_id,
                        candidate_rank=rank,
                        constraint_id=constraint_id,
                        binding_state=BindingState.UNRESOLVED,
                        is_contextual=False,
                        is_discriminative=False,
                        evidence_image_indices=(),
                        evidence_ids=(),
                        source_sha256=commitment,
                        verified=False,
                        verification_source="PENDING_INDEPENDENT_OPTION_BINDING_REVIEW",
                    )
                    edges.append(edge)
                    event_edges.append({
                        "prefix_step": step,
                        "option_id": option_id,
                        "candidate_id": candidate_id,
                        "candidate_rank": rank,
                        "constraint_id": constraint_id,
                        "binding_state": "UNRESOLVED",
                        "verified": False,
                        "review_status": "PENDING_INDEPENDENT_OPTION_BINDING_REVIEW",
                        "source_sha256": commitment,
                    })
        review_events.append({
            "event_id": event_id,
            "dataset": dataset,
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "instruction": str(row["instruction"]),
            "decision_step": decision_step,
            "current_candidate_ids": list(option_aliases),
            "constraint_graph": graph,
            "constraint_roles": roles,
            "independent_missing_constraint_count": len(missing),
            "prefixes": prefixes,
            "candidate_id_source": "sealed_opaque_causal_alias_only",
            "candidate_rank_is_semantic_truth": False,
            "option_identities": [identity.as_mapping() for identity in event_identities],
            "binding_review_rows": event_edges,
            "binding_review_required": True,
            "visual_review_source": "MF3ZP_CODEX_VISUAL_REVIEW_LABELS; no option binding present",
        })
    review_events.sort(key=lambda item: (item["dataset"], item["scene_id"], item["episode_id"], item["event_id"]))
    source = {
        "schema_version": "revealnav-mf3zr-option-binding-review-source/1",
        "status": "OPTION_BINDING_REVIEW_SOURCE_ONLY",
        "revision": "mf3zr_option_bound_support_v1",
        "events": review_events,
        "event_count": len(review_events),
        "no_replacement": True,
        "outcome_blind": True,
        "qwen_calls": 0,
        "qwen_reads": 0,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "human_binding_review_completed": False,
    }
    diagnostics = {
        "events": len(review_events),
        "option_identities": len(identities),
        "provisional_edges": len(edges),
        "unsupported_reason_counts": dict(sorted(reason_counts.items())),
        "unverified_edges": len(edges),
    }
    return source, tuple(identities), tuple(edges), diagnostics


def audit_bindings(
    *,
    review_source: Mapping[str, object],
    bindings: Sequence[OptionEvidenceBinding],
    identities: Sequence[OptionIdentity],
) -> dict[str, object]:
    """Audit completeness; no edge is accepted without independent review."""

    if review_source.get("outcome_payload_read") is not False:
        raise BindingAuditError("review source outcome flag opened")
    by_event: dict[str, Mapping[str, object]] = {str(row["event_id"]): row for row in review_source.get("events", ())}
    identity_by_event = {}
    for identity in identities:
        identity_by_event.setdefault(identity.event_id, []).append(identity)
    accepted = 0
    reasons: Counter[str] = Counter()
    event_records = []
    for event_id, event in sorted(by_event.items()):
        roles = {str(k): str(v) for k, v in event.get("constraint_roles", {}).items()}
        required = {cid for cid, role in roles.items() if role in ACTIVE_ROLES}
        event_bindings = [edge for edge in bindings if edge.event_id == event_id]
        verified = [edge for edge in event_bindings if edge.usable]
        missing = [cid for cid in required if not any(edge.constraint_id == cid for edge in verified)]
        event_reasons: list[str] = []
        if any(identity.identity_status == "OPTION_IDENTITY_UNRESOLVED" for identity in identity_by_event.get(event_id, ())):
            event_reasons.append("UNSUPPORTED_OPTION_IDENTITY")
        if missing:
            event_reasons.append("UNSUPPORTED_UNVERIFIED_OPTION_BINDING")
        if event.get("independent_missing_constraint_count", 0):
            event_reasons.append("UNSUPPORTED_MISSING_DEC_BINDING")
        if not event_reasons:
            accepted += 1
        else:
            reasons.update(event_reasons)
        event_records.append({
            "event_id": event_id,
            "dataset": event["dataset"],
            "scene_id": event["scene_id"],
            "episode_id": event["episode_id"],
            "required_active_constraints": len(required),
            "verified_binding_edges": len(verified),
            "missing_active_constraints": sorted(missing),
            "status": "VALID_OPTION_BINDING" if not event_reasons else "UNSUPPORTED_OPTION_BINDING",
            "unsupported_reasons": sorted(set(event_reasons)),
        })
    return {
        "schema_version": "revealnav-mf3zr-option-binding-audit/1",
        "status": "MF3ZR_BINDING_AUDIT_COMPLETE",
        "events": len(by_event),
        "valid_option_binding_events": accepted,
        "unsupported_events": len(by_event) - accepted,
        "unsupported_reason_counts": dict(sorted(reasons.items())),
        "event_records": event_records,
        "binding_state_counts": dict(sorted(Counter(edge.binding_state.value for edge in bindings).items())),
        "verification_source_counts": dict(sorted(Counter(edge.verification_source for edge in bindings).items())),
        "human_binding_review_completed": False,
        "outcome_payload_read": False,
        "qwen_calls": 0,
        "qwen_reads": 0,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }


__all__ = [
    "ROLE_DEC", "ROLE_PRE", "ACTIVE_ROLES", "BindingAuditError",
    "build_review_source", "audit_bindings",
]
