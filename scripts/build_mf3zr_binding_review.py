#!/usr/bin/env python3
"""Build the MF3ZR option-binding review source.

This command only materializes outcome-blind references for the already sealed
80-event MF3ZQ population.  It does not call Qwen, read metrics, or assign a
semantic binding state.
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

from revealnav_mf3.option_binding_audit import BindingAuditError, build_review_source  # noqa: E402
from revealnav_mf3.mf3zr_protocol import (  # noqa: E402
    OUTPUT,
    REVIEW_SOURCE_PATH,
    SOURCE_POPULATION,
    SOURCE_VISUAL_LABELS,
    SOURCE_POPULATION_SHA,
    SOURCE_VISUAL_LABEL_SHA,
    sha256_file,
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise BindingAuditError(f"expected object in {path}")
            rows.append(dict(value))
    return rows


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise BindingAuditError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise BindingAuditError(f"stale partial artifact: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def build() -> dict[str, object]:
    if sha256_file(SOURCE_POPULATION) != SOURCE_POPULATION_SHA:
        raise BindingAuditError("fixed MF3ZQ population hash mismatch")
    if sha256_file(SOURCE_VISUAL_LABELS) != SOURCE_VISUAL_LABEL_SHA:
        raise BindingAuditError("fixed visual-label hash mismatch")
    population = _jsonl(SOURCE_POPULATION)
    labels = _jsonl(SOURCE_VISUAL_LABELS)
    if len(population) != 80 or len(labels) != 80:
        raise BindingAuditError("MF3ZR requires exactly 80 fixed rows")
    source, identities, edges, diagnostics = build_review_source(
        root=ROOT,
        population_rows=population,
        label_rows=labels,
    )
    _write_new(REVIEW_SOURCE_PATH, source)
    # This summary is intentionally not a scientific support result; it is a
    # source-materialization receipt and contains no outcome fields.
    receipt = {
        "schema_version": "revealnav-mf3zr-binding-review-source-receipt/1",
        "status": "MF3ZR_BINDING_REVIEW_SOURCE_READY",
        "source_population_sha256": sha256_file(SOURCE_POPULATION),
        "source_visual_label_sha256": sha256_file(SOURCE_VISUAL_LABELS),
        "review_source_sha256": sha256_file(REVIEW_SOURCE_PATH),
        "events": len(population),
        "unique_episodes": len({(str(row["dataset"]), str(row["episode_id"])) for row in population}),
        "raw_mp3d_scenes": len({str(row["scene_id"]) for row in population}),
        "domain_counts": {"R2R": sum(str(row["dataset"]) == "R2R" for row in population), "RxR": sum(str(row["dataset"]) == "RxR" for row in population)},
        "option_identities": len(identities),
        "provisional_binding_edges": len(edges),
        "diagnostics": diagnostics,
        "binding_review_completed": False,
        "qwen_calls": 0,
        "qwen_reads": 0,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    receipt_path = OUTPUT / "MF3ZR_OPTION_BINDING_REVIEW_SOURCE_RECEIPT.json"
    _write_new(receipt_path, receipt)
    return receipt


def main() -> int:
    try:
        result = build()
    except (OSError, KeyError, TypeError, ValueError, BindingAuditError) as error:
        print(f"MF3ZR_REVIEW_SOURCE_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
