#!/usr/bin/env python3
"""Compute the MF3ZR option-bound support gate.

Only causal identity, annotation-role, and control-availability metadata are
read.  Navigation metrics and historical outcomes are deliberately not
opened.  With the current source, unverified bindings and an unavailable
frozen-controller callback cause a deterministic ``SUPPORT_FAIL``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zr_protocol import (  # noqa: E402
    BINDING_AUDIT_PATH,
    BINDINGS_PATH,
    DOMAIN_COVERAGE_MIN,
    JOINT_AUDIT_PATH,
    JOINT_COVERAGE_MIN,
    MIN_SUPPORTED_EPISODES_PER_DOMAIN,
    OUTPUT,
    PROTOCOL_PATH,
    RESULT_PATH,
    RETURNABILITY_AUDIT_PATH,
    RETURNABILITY_PATH,
    REVEAL_EXPIRY_PATH,
    REVIEW_SOURCE_PATH,
    RETURN_HORIZON,
    verify_protocol,
)
from revealnav_mf3.reveal_expiry_support import (  # noqa: E402
    ExpirySupportStatus,
    RevealSupportStatus,
)


class SupportAuditError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SupportAuditError(f"expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise SupportAuditError(f"expected JSON object in {path}")
            result.append(dict(value))
    return result


def _write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise SupportAuditError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise SupportAuditError(f"stale partial artifact: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_diag(event: Mapping[str, object]) -> dict[str, object]:
    roles = {str(key): str(value) for key, value in event.get("constraint_roles", {}).items()}
    lengths = [len(str(event.get("instruction", "")).split()), len(roles)]
    return {
        "event_id": str(event["event_id"]),
        "dataset": str(event["dataset"]),
        "scene_id": str(event["scene_id"]),
        "instruction_token_length": lengths[0],
        "total_constraint_count": lengths[1],
        "dec_required_count": sum(role == "DEC_REQUIRED" for role in roles.values()),
        "prerequisite_count": sum(role == "PREREQUISITE_ONLY" for role in roles.values()),
        "candidate_count": len(event.get("current_candidate_ids", ())),
        "prefix_count": len(event.get("prefixes", ())),
        "decision_step": int(event["decision_step"]),
        "trigger_type": "not_available_in_sealed_population",
    }


def _summarize_distribution(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    numeric = (
        "instruction_token_length", "total_constraint_count", "dec_required_count",
        "prerequisite_count", "candidate_count", "prefix_count", "decision_step",
    )
    output: dict[str, object] = {"count": len(rows)}
    for field in numeric:
        values = [int(row[field]) for row in rows]
        output[field] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    output["domain_counts"] = dict(sorted(Counter(str(row["dataset"]) for row in rows).items()))
    output["scene_count"] = len({str(row["scene_id"]) for row in rows})
    return output


def run() -> dict[str, object]:
    verify_protocol(PROTOCOL_PATH)
    source = _read_json(REVIEW_SOURCE_PATH)
    binding_audit = _read_json(BINDING_AUDIT_PATH)
    returnability_audit = _read_json(RETURNABILITY_AUDIT_PATH)
    binding_rows = _read_jsonl(BINDINGS_PATH)
    return_rows = _read_jsonl(RETURNABILITY_PATH)
    events = [dict(event) for event in source.get("events", ()) if isinstance(event, Mapping)]
    if len(events) != 80:
        raise SupportAuditError("MF3ZR support denominator drift")
    return_by_option: dict[str, list[dict[str, object]]] = {}
    for row in return_rows:
        return_by_option.setdefault(str(row["option_id"]), []).append(row)
    # One support row per option identity.  Missing binding or control support
    # is represented explicitly; no option is replaced or silently dropped.
    reveal_expiry_rows: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []
    supported_events: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    option_count = 0
    binding_state_counts: Counter[str] = Counter()
    for event in events:
        event_id = str(event["event_id"])
        identities = [item for item in event.get("option_identities", ()) if isinstance(item, Mapping)]
        option_count += len(identities)
        event_reasons: set[str] = set()
        required_count = sum(str(role) in {"DEC_REQUIRED", "PREREQUISITE_ONLY"} for role in event.get("constraint_roles", {}).values())
        for edge in binding_rows:
            if str(edge.get("event_id")) == event_id:
                binding_state_counts[str(edge.get("binding_state"))] += 1
        for identity in identities:
            option_id = str(identity["option_id"])
            option_returns = return_by_option.get(option_id, [])
            return_available = bool(option_returns) and all(str(item.get("status")) not in {"EXECUTION_UNAVAILABLE", "INVALID_ANCHOR"} for item in option_returns)
            reveal_status = RevealSupportStatus.REVEAL_NOT_COMPUTABLE
            expiry_status = ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE
            reveal_reason = "OPTION_BINDING_NOT_VERIFIED"
            expiry_reason = "CONTROL_RETURNABILITY_UNAVAILABLE"
            if not return_available:
                event_reasons.add("UNSUPPORTED_RETURNABILITY")
            if str(identity.get("identity_status")) == "OPTION_IDENTITY_UNRESOLVED":
                event_reasons.add("UNSUPPORTED_OPTION_IDENTITY")
            event_reasons.add("UNSUPPORTED_UNVERIFIED_OPTION_BINDING")
            reveal_expiry_rows.append({
                "schema_version": "revealnav-mf3zr-reveal-expiry-support/1",
                "event_id": event_id,
                "option_id": option_id,
                "reveal": {"status": reveal_status.value, "step": None, "reason": reveal_reason},
                "expiry": {"status": expiry_status.value, "step": None, "reason": expiry_reason},
                "computable": False,
                "returnability_statuses": sorted({str(item.get("status")) for item in option_returns}),
            })
        if int(event.get("independent_missing_constraint_count", 0)) > 0:
            event_reasons.add("UNSUPPORTED_MISSING_DEC_BINDING")
        if not identities:
            event_reasons.add("UNSUPPORTED_OPTION_IDENTITY")
        status = "SUPPORTED" if not event_reasons else "UNSUPPORTED_MULTIPLE" if len(event_reasons) > 1 else next(iter(event_reasons))
        if status == "SUPPORTED":
            supported_events.append(event)
        else:
            reasons.update(event_reasons)
        event_records.append({
            "event_id": event_id,
            "dataset": str(event["dataset"]),
            "scene_id": str(event["scene_id"]),
            "episode_id": str(event["episode_id"]),
            "option_count": len(identities),
            "required_active_constraint_count": required_count,
            "status": status,
            "unsupported_reasons": sorted(event_reasons),
        })
    _write_new(REVEAL_EXPIRY_PATH, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in reveal_expiry_rows))
    supported_by_domain = Counter(str(row["dataset"]) for row in supported_events)
    total_by_domain = Counter(str(event["dataset"]) for event in events)
    domain_coverage = {domain: (supported_by_domain[domain] / total_by_domain[domain] if total_by_domain[domain] else 0.0) for domain in ("R2R", "RxR")}
    diagnostics = [_safe_diag(event) for event in events]
    supported_diag = [row for row, event in zip(diagnostics, events) if event in supported_events]
    unsupported_diag = [row for row, event in zip(diagnostics, events) if event not in supported_events]
    joint_audit = {
        "schema_version": "revealnav-mf3zr-joint-support-audit/1",
        "status": "MF3ZR_JOINT_SUPPORT_AUDIT_COMPLETE",
        "events": len(events),
        "unique_episodes": len({(str(event["dataset"]), str(event["episode_id"])) for event in events}),
        "raw_mp3d_scenes": len({str(event["scene_id"]) for event in events}),
        "binding_audit_status": binding_audit.get("status"),
        "returnability_audit_status": returnability_audit.get("status"),
        "binding_state_counts": dict(sorted(binding_state_counts.items())),
        "supported_events": len(supported_events),
        "unsupported_events": len(events) - len(supported_events),
        "unsupported_reason_counts": dict(sorted(reasons.items())),
        "domain_coverage": domain_coverage,
        "domain_supported_unique_episodes": {domain: len({str(event["episode_id"]) for event in supported_events if str(event["dataset"]) == domain}) for domain in ("R2R", "RxR")},
        "coverage_thresholds": {"joint": JOINT_COVERAGE_MIN, "domain": DOMAIN_COVERAGE_MIN, "minimum_episodes_per_domain": MIN_SUPPORTED_EPISODES_PER_DOMAIN},
        "event_records": event_records,
        "outcome_blind_diagnostics": {
            "supported": _summarize_distribution(supported_diag),
            "unsupported": _summarize_distribution(unsupported_diag),
            "fields": ["domain", "raw_scene", "instruction_token_length", "total_constraint_count", "dec_required_count", "prerequisite_count", "candidate_count", "prefix_count", "decision_step", "trigger_type"],
        },
        "outcome_payload_read": False,
        "qwen_calls": 0,
        "qwen_reads": 0,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "oracle_arms_run": [],
        "checkpoint_generated": False,
    }
    _write_new(JOINT_AUDIT_PATH, json.dumps(joint_audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    pass_gate = (
        len(supported_events) / len(events) >= JOINT_COVERAGE_MIN
        and all(domain_coverage[domain] >= DOMAIN_COVERAGE_MIN for domain in ("R2R", "RxR"))
        and all(len({str(event["episode_id"]) for event in supported_events if str(event["dataset"]) == domain}) >= MIN_SUPPORTED_EPISODES_PER_DOMAIN for domain in ("R2R", "RxR"))
    )
    result = {
        "schema_version": "revealnav-mf3zr-option-bound-support-result/1",
        "revision": "mf3zr_option_bound_support_v1",
        "status": "SUPPORT_PASS" if pass_gate else "SUPPORT_FAIL",
        "signal_status": "OPTION_BOUND_SUPPORT_AVAILABLE" if pass_gate else "OPTION_BOUND_SUPPORT_INSUFFICIENT",
        "source_population_sha256": _sha256(Path(str(ROOT / "artifacts/training/mf3zq_oracle_revealskill_headroom_v1/MF3ZQ_ORACLE_HEADROOM_POPULATION.jsonl"))),
        "source_visual_label_sha256": _sha256(Path(str(ROOT / "artifacts/training/mf3zp_codex_visual_review_v1/MF3ZP_CODEX_VISUAL_REVIEW_LABELS.jsonl"))),
        "events": len(events),
        "unique_episodes": len({(str(event["dataset"]), str(event["episode_id"])) for event in events}),
        "raw_mp3d_scenes": len({str(event["scene_id"]) for event in events}),
        "domain_counts": dict(sorted(total_by_domain.items())),
        "binding_supported": int(binding_audit.get("valid_option_binding_events", 0)),
        "returnability_supported": int(returnability_audit.get("success_count", 0)),
        "reveal_supported": 0,
        "expiry_supported": 0,
        "joint_supported": len(supported_events),
        "joint_coverage": len(supported_events) / len(events),
        "R2R_joint_coverage": domain_coverage["R2R"],
        "RxR_joint_coverage": domain_coverage["RxR"],
        "domain_supported_unique_episodes": joint_audit["domain_supported_unique_episodes"],
        "unsupported_reason_counts": dict(sorted(reasons.items())),
        "option_count": option_count,
        "binding_edge_count": len(binding_rows),
        "binding_state_counts": dict(sorted(binding_state_counts.items())),
        "returnability_callback_available": bool(returnability_audit.get("callback_available")),
        "returnability_status_counts": returnability_audit.get("status_counts", {}),
        "reveal_expiry_status": {"reveal_not_computable": len(reveal_expiry_rows), "expiry_not_computable": len(reveal_expiry_rows)},
        "support_coverage_gate": {"joint_min": JOINT_COVERAGE_MIN, "domain_min": DOMAIN_COVERAGE_MIN, "episodes_per_domain_min": MIN_SUPPORTED_EPISODES_PER_DOMAIN},
        "outcome_blind_diagnostics": joint_audit["outcome_blind_diagnostics"],
        "qwen_calls": 0,
        "qwen_reads": 0,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "checkpoint_generated": False,
        "oracle_arms_run": [],
        "oracle_numerical_evidence": "NOT_OBSERVED",
        "downstream_oracle_authorized": False,
        "ready_for_separately_versioned_mf3zs_oracle_protocol": pass_gate,
        "failure_status_if_not_pass": "MF3ZR_OPTION_BOUND_SUPPORT_FAIL",
    }
    _write_new(RESULT_PATH, json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    try:
        result = run()
    except (OSError, KeyError, TypeError, ValueError, SupportAuditError, RuntimeError) as error:
        print(f"MF3ZR_SUPPORT_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") == "SUPPORT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
