#!/usr/bin/env python3
"""Run the one-shot MF3ZQ Oracle RevealSkill headroom audit.

The command intentionally performs the support gate before opening any task
metrics.  If the sealed population cannot provide option-specific DEC chains
and a control-backed returnability implementation, it writes a fail-closed
diagnostic and stops; it never guesses an option or fabricates arm results.
When a future, independently sealed population satisfies the support gate, the
same entry point is the only place allowed to dispatch matched frozen-controller
workers.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.oracle_headroom_metrics import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATASTROPHIC_THRESHOLD,
)
from revealnav_mf3.oracle_headroom_protocol import (  # noqa: E402
    AUDIT_PATH,
    OUTPUT,
    POPULATION_PATH,
    PROTOCOL_PATH,
    PUBLIC_CLOSED,
    RESULT_PATH,
    ProtocolError,
    sha256_file,
    verify_protocol,
)


class HeadroomRunError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HeadroomRunError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HeadroomRunError(f"expected JSON object in {path}")
            rows.append(value)
    if not rows:
        raise HeadroomRunError(f"empty population: {path}")
    return rows


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise HeadroomRunError(f"refusing to overwrite immutable result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise HeadroomRunError(f"stale result partial: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _verify_population(protocol: Mapping[str, object]) -> list[dict]:
    inventory = protocol["source_files"]["population"]
    if sha256_file(POPULATION_PATH) != inventory["sha256"] or POPULATION_PATH.stat().st_size != int(inventory["bytes"]):
        raise HeadroomRunError("sealed MF3ZQ population inventory drift")
    rows = _read_jsonl(POPULATION_PATH)
    if len(rows) != 80:
        raise HeadroomRunError("MF3ZQ population count drift")
    identities = {(str(row["dataset"]), str(row["episode_id"])) for row in rows}
    if len(identities) != 80:
        raise HeadroomRunError("population is not one-event-per-episode")
    if Counter(str(row["dataset"]) for row in rows) != Counter({"R2R": 40, "RxR": 40}):
        raise HeadroomRunError("population domain balance drift")
    for row in rows:
        if row.get("support", {}).get("legal_oracle_continuation_supported") is not False:
            raise HeadroomRunError("population support flag is not explicit")
        steps = [int(value) for value in row.get("prefix_steps", [])]
        if not steps or steps[-1] != int(row["decision_step"]) or any(value > int(row["decision_step"]) for value in steps):
            raise HeadroomRunError("future/non-causal prefix entered population")
    return rows


def _decomposition(rows: list[dict]) -> dict[str, object]:
    counts = Counter(str(role) for row in rows for role in row.get("constraint_roles", {}).values())
    total = sum(counts.values())
    return {
        "DEC_REQUIRED": counts["DEC_REQUIRED"],
        "PREREQUISITE_ONLY": counts["PREREQUISITE_ONLY"],
        "FUTURE_NOT_RELEVANT": counts["FUTURE_NOT_RELEVANT"],
        "INCORRECT": counts["INCORRECT"],
        "REDUNDANT": counts["REDUNDANT"],
        "missing_DEC_constraints": sum(len(row.get("independent_missing_constraints", [])) for row in rows),
        "active_decisive_fraction": counts["DEC_REQUIRED"] / total if total else 0.0,
        "current_relevant_fraction": (counts["DEC_REQUIRED"] + counts["PREREQUISITE_ONLY"]) / total if total else 0.0,
    }


def _unsupported(rows: list[dict]) -> tuple[list[dict], Counter]:
    unsupported = [row for row in rows if not row["support"]["legal_oracle_continuation_supported"]]
    reasons = Counter(reason for row in unsupported for reason in row["support"].get("unsupported_reasons", []))
    return unsupported, reasons


def _failure_result(protocol: Mapping[str, object], rows: list[dict], unsupported: list[dict], reasons: Counter) -> dict[str, object]:
    domains = {}
    for domain in ("R2R", "RxR"):
        subset = [row for row in rows if row["dataset"] == domain]
        bad = [row for row in unsupported if row["dataset"] == domain]
        domains[domain] = {
            "events": len(subset),
            "unique_episodes": len({str(row["episode_id"]) for row in subset}),
            "raw_scenes": len({str(row["scene_id"]) for row in subset}),
            "unsupported_episodes": len(bad),
            "unsupported_event_ids": [str(row["event_id"]) for row in bad],
            "baseline": {"status": "NOT_RUN", "reason": "support gate failed before metrics"},
            "oracle_dec": {"status": "NOT_RUN"},
            "oracle_dec_option_memory": {"status": "NOT_RUN"},
            "full_oracle_revealskill": {"status": "NOT_RUN"},
        }
    return {
        "schema_version": "revealnav-mf3zq-oracle-headroom-result/1",
        "revision": "mf3zq_oracle_revealskill_headroom_v1",
        "status": "MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL",
        "signal_status": "UNSUPPORTED_ORACLE_POPULATION",
        "first_failure": "population support audit",
        "failure_reason": "No option-specific DEC binding and no control-backed returnability are available in the sealed independent visual-review population; no option or action was guessed.",
        "stop_rule_triggered": True,
        "downstream_rollouts_started": False,
        "arms_run": [],
        "episodes": len(rows),
        "events": len(rows),
        "unique_episodes": len({(str(row["dataset"]), str(row["episode_id"])) for row in rows}),
        "raw_mp3d_scenes": len({str(row["scene_id"]) for row in rows}),
        "domain_counts": {"R2R": sum(row["dataset"] == "R2R" for row in rows), "RxR": sum(row["dataset"] == "RxR" for row in rows)},
        "unsupported_episode_count": len(unsupported),
        "unsupported_reasons": dict(sorted(reasons.items())),
        "domains": domains,
        "decomposition": _decomposition(rows),
        "utility": {"nDTW": 0.50, "SDTW": 0.25, "SPL": 0.25},
        "catastrophic_delta_utility": CATASTROPHIC_THRESHOLD,
        "bootstrap": {"cluster": "raw_mp3d_scene", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "qwen": {"reads": 0, "calls": 0},
        "human_labels_fabricated": False,
        "checkpoint_generated": False,
        "public_split_access": dict(PUBLIC_CLOSED),
        "formal_mf3zp_protocol_unchanged": True,
        "formal_mf3zp_oracle_headroom_authorized": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_label_sha256": protocol["source_files"]["visual_labels"]["sha256"],
        "source_population_sha256": protocol["source_files"]["population"]["sha256"],
        "metrics_read_before_action": False,
        "generated_at_epoch_s": time.time(),
    }


def run() -> dict[str, object]:
    protocol = verify_protocol(PROTOCOL_PATH)
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise HeadroomRunError("MF3ZQ result already exists; one-shot rerun is forbidden")
    rows = _verify_population(protocol)
    unsupported, reasons = _unsupported(rows)
    # The support gate is deliberately before any task metric read or worker
    # dispatch.  This population has no explicit candidate-to-DEC mapping and
    # no executable returnability callback, so no arm can be honestly run.
    if unsupported:
        result = _failure_result(protocol, rows, unsupported, reasons)
        _atomic_json(RESULT_PATH, result)
        return result
    raise HeadroomRunError(
        "MF3ZQ population unexpectedly has complete support; independent implementation review is required before dispatching rollouts"
    )


def main() -> int:
    try:
        result = run()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError, HeadroomRunError) as error:
        print(f"MF3ZQ_RUN_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 2 if result.get("status") == "MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
