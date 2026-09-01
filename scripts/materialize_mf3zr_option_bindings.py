#!/usr/bin/env python3
"""Materialize conservative MF3ZR option-binding records.

The fixed source has no independently verified candidate-specific labels.  The
script preserves that fact with explicit ``UNRESOLVED``/``verified=false``
edges and writes an audit that cannot authorize downstream computation.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.option_binding_audit import BindingAuditError, audit_bindings  # noqa: E402
from revealnav_mf3.option_binding_schema import BindingState, OptionEvidenceBinding, OptionIdentity  # noqa: E402
from revealnav_mf3.mf3zr_protocol import (  # noqa: E402
    BINDING_AUDIT_PATH,
    BINDINGS_PATH,
    PROTOCOL_PATH,
    REVIEW_SOURCE_PATH,
    verify_protocol,
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BindingAuditError(f"expected object: {path}")
    return dict(value)


def _write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise BindingAuditError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise BindingAuditError(f"stale partial artifact: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _load_edges(source: Mapping[str, object]) -> tuple[tuple[OptionIdentity, ...], tuple[OptionEvidenceBinding, ...]]:
    identities: list[OptionIdentity] = []
    edges: list[OptionEvidenceBinding] = []
    for event in source.get("events", ()):
        if not isinstance(event, Mapping):
            raise BindingAuditError("review source event is malformed")
        for raw_identity in event.get("option_identities", ()):
            if not isinstance(raw_identity, Mapping):
                raise BindingAuditError("option identity row is malformed")
            identities.append(OptionIdentity(
                option_id=str(raw_identity["option_id"]),
                event_id=str(raw_identity["event_id"]),
                candidate_id=str(raw_identity["candidate_id"]),
                first_seen_step=int(raw_identity["first_seen_step"]),
                last_seen_step=int(raw_identity["last_seen_step"]),
                anchor_checkpoint_id=str(raw_identity["anchor_checkpoint_id"]),
                identity_status=str(raw_identity["identity_status"]),
            ))
        for raw_edge in event.get("binding_review_rows", ()):
            if not isinstance(raw_edge, Mapping):
                raise BindingAuditError("binding review row is malformed")
            edges.append(OptionEvidenceBinding(
                event_id=str(event["event_id"]),
                prefix_step=int(raw_edge["prefix_step"]),
                option_id=str(raw_edge["option_id"]),
                candidate_id=str(raw_edge["candidate_id"]),
                candidate_rank=int(raw_edge["candidate_rank"]),
                constraint_id=str(raw_edge["constraint_id"]),
                binding_state=BindingState(str(raw_edge["binding_state"])),
                is_contextual=False,
                is_discriminative=False,
                evidence_image_indices=(),
                evidence_ids=(),
                source_sha256=str(raw_edge["source_sha256"]),
                verified=False,
                verification_source="PENDING_INDEPENDENT_OPTION_BINDING_REVIEW",
            ))
    return tuple(identities), tuple(edges)


def materialize() -> dict[str, object]:
    verify_protocol(PROTOCOL_PATH)
    if not REVIEW_SOURCE_PATH.is_file():
        raise BindingAuditError("MF3ZR binding review source is missing")
    source = _read_json(REVIEW_SOURCE_PATH)
    identities, edges = _load_edges(source)
    lines = "".join(json.dumps(edge.as_mapping(), sort_keys=True, ensure_ascii=False) + "\n" for edge in edges)
    _write_new(BINDINGS_PATH, lines)
    audit = audit_bindings(review_source=source, bindings=edges, identities=identities)
    audit["source_review_sha256"] = __import__("hashlib").sha256(REVIEW_SOURCE_PATH.read_bytes()).hexdigest()
    audit["option_identity_count"] = len(identities)
    audit["binding_edge_count"] = len(edges)
    _write_new(BINDING_AUDIT_PATH, json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return audit


def main() -> int:
    try:
        result = materialize()
    except (OSError, KeyError, TypeError, ValueError, BindingAuditError, RuntimeError) as error:
        print(f"MF3ZR_BINDING_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
