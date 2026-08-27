#!/usr/bin/env python3
"""Join validated human decisions with reproducible resource-conditioned T_X.

The result is an engineering seed set.  One reviewer cannot create the frozen
three-reviewer pilot or final Gold benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
PACKAGE = BASE / "human_pilot_300"
SELECTION = PACKAGE / "RXR_HUMAN_PILOT_300_SELECTION.json"
LABELS = PACKAGE / "daiyang_rxr300.jsonl"
LABEL_ACCEPTANCE = PACKAGE / "RXR_HUMAN_PILOT_300_LABEL_ACCEPTANCE.json"
TX_GATE = BASE / "tx_gate/RXR_EXPANSION_TX_GATE.json"
DEFAULT_OUTPUT = BASE / "RXR_SINGLE_REVIEW_HUMAN_TX_JOIN.json"
FROZEN = ROOT / "FROZEN_SPEC.md"
PROTOCOL = ROOT / "PHASE0_PROTOCOL.md"
EXPECTED_SELECTION_SHA256 = (
    "c8d79f8aa7285b050568759d5f492ff288186520551749e6a5f10a2c55dfd179"
)
EXPECTED_FROZEN_SHA256 = (
    "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81"
)
EXPECTED_PROTOCOL_SHA256 = (
    "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    project = ROOT.resolve()
    if project not in output.parents:
        raise SystemExit("output must resolve inside the project")
    required = [SELECTION, LABELS, LABEL_ACCEPTANCE, TX_GATE, FROZEN, PROTOCOL]
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"required regular non-symlink file missing: {path}")
    expected_static = {
        SELECTION: EXPECTED_SELECTION_SHA256,
        FROZEN: EXPECTED_FROZEN_SHA256,
        PROTOCOL: EXPECTED_PROTOCOL_SHA256,
    }
    for path, expected in expected_static.items():
        if sha256_file(path) != expected:
            raise SystemExit(f"frozen source drift: {path}")

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    acceptance = json.loads(LABEL_ACCEPTANCE.read_text(encoding="utf-8"))
    tx_gate = json.loads(TX_GATE.read_text(encoding="utf-8"))
    labels = read_jsonl(LABELS)
    label_sha = sha256_file(LABELS)
    acceptance_sources = acceptance.get("sources", {})
    label_relative = str(LABELS.relative_to(ROOT))
    preconditions = {
        "human_label_acceptance_pass": acceptance.get("status")
            == "HUMAN_LABELS_PASS_TX_JOIN_REQUIRED",
        "human_label_file_hash_bound": acceptance_sources.get(label_relative)
            == label_sha,
        "tx_expansion_pass": tx_gate.get("status")
            == "TX_EXPANSION_PASS_HUMAN_JOIN_REQUIRED",
        "tx_all_525_complete": tx_gate.get("gates", {}).get(
            "all_525_complete") is True,
        "tx_all_525_exactly_reproduced": tx_gate.get("gates", {}).get(
            "all_525_exactly_reproduced") is True,
        "selection_exact": sha256_file(SELECTION)
            == EXPECTED_SELECTION_SHA256,
    }
    if not all(preconditions.values()):
        raise SystemExit(
            "join precondition failed: "
            + json.dumps(preconditions, sort_keys=True)
        )

    selected = selection["items"]
    selected_ids = [row["event_id"] for row in selected]
    labels_by_id = {row["event_id"]: row for row in labels}
    tx_by_id = {row["event_id"]: row for row in tx_gate["events"]}
    if (len(labels) != 300 or len(labels_by_id) != 300
            or set(labels_by_id) != set(selected_ids)):
        raise SystemExit("validated label rows do not match selection")
    if not set(selected_ids) <= set(tx_by_id):
        raise SystemExit("one or more selected events lack T_X evidence")

    joined = []
    dispositions = Counter()
    positive_scenes = set()
    core_positive = 0
    for item in selected:
        event_id = item["event_id"]
        label = labels_by_id[event_id]
        tx = tx_by_id[event_id]
        if label["final_label"] == "ACCEPT" and tx[
                "passes_frozen_two_budget_gate"]:
            disposition = "SINGLE_REVIEW_ACCEPT_TX_ADMITTED"
            positive_scenes.add(item["scene_id"])
            if item["cohort"] == "AUDIT_CORE_UNIFORM_250":
                core_positive += 1
        elif label["final_label"] == "ACCEPT":
            disposition = "SINGLE_REVIEW_ACCEPT_TX_NOT_ADMITTED"
        elif label["final_label"] == "REJECT":
            disposition = "HUMAN_REJECT"
        else:
            disposition = "HUMAN_AMBIGUOUS"
        dispositions[disposition] += 1
        joined.append({
            "review_index": item["review_index"],
            "cohort": item["cohort"],
            "event_id": event_id,
            "episode_id": item["episode_id"],
            "scene_id": item["scene_id"],
            "human_final_label": label["final_label"],
            "human_reason_codes": label["reason_codes"],
            "tx_admitted": tx["passes_frozen_two_budget_gate"],
            "disposition": disposition,
            "strict_reveal_interval": tx["strict_reveal_interval"],
            "observed_prefix_horizon": tx["observed_prefix_horizon"],
            "frontier_status": tx["frontier_status"],
            "frozen_unique_last_safe_budget_count": tx[
                "frozen_unique_last_safe_budget_count"],
            "event_evidence_sha256": tx["event_evidence_sha256"],
            "independent_process_exact_reproduction": tx[
                "independent_process_exact_reproduction"],
            "round1": tx["round1"],
            "round2": tx["round2"],
        })

    positive_count = dispositions["SINGLE_REVIEW_ACCEPT_TX_ADMITTED"]
    engineering_seed_gates = {
        "all_300_human_decisions_joined": len(joined) == 300,
        "all_joined_tx_exactly_reproduced": all(
            row["independent_process_exact_reproduction"] for row in joined
        ),
        "single_review_positive_fraction_at_least_25pct":
            positive_count / 300 >= 0.25,
        "single_review_positive_scene_count_at_least_25":
            len(positive_scenes) >= 25,
    }
    output_value = {
        "manifest": "RevealNav RxR single-review human and T_X join",
        "revision": "rxr-single-review-human-tx-join/1",
        "status": "ENGINEERING_SEED_PASS_THREE_REVIEWER_PILOT_REQUIRED"
            if all(engineering_seed_gates.values())
            else "ENGINEERING_SEED_FAIL",
        "scope": (
            "single-reviewer engineering seed; explicitly not the frozen "
            "three-reviewer pilot, Gold test, or paper benchmark"
        ),
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in required
        },
        "preconditions": preconditions,
        "counts": {
            "joined_events": len(joined),
            "dispositions": dict(sorted(dispositions.items())),
            "single_review_tx_positive_events": positive_count,
            "single_review_tx_positive_fraction": round(
                positive_count / 300, 6),
            "single_review_tx_positive_scenes": len(positive_scenes),
            "core_250_single_review_tx_positive_events": core_positive,
        },
        "engineering_seed_gates": engineering_seed_gates,
        "events": joined,
        "frozen_submission_gates_not_yet_satisfied": [
            "300-event three-person annotation",
            "U/A/D Fleiss kappa >= 0.65",
            "evidence-closure kappa >= 0.70",
            "at least 2000 scene-disjoint Reveal Events",
            "Gold test with at least 600 three-person annotated events",
        ],
        "three_reviewer_agreement_measured": False,
        "gold_test_created": False,
        "paper_benchmark_claim_authorized": False,
        "full_training_authorized": False,
        "exploratory_training_seed_available": all(
            engineering_seed_gates.values()),
        "future_suffix_used_only_for_offline_tx_label": True,
        "online_future_information_used": 0,
    }
    atomic_json(output, output_value)
    print(json.dumps({
        "status": output_value["status"],
        "counts": output_value["counts"],
        "gates": engineering_seed_gates,
        "output": str(output.relative_to(project)),
    }, indent=2, ensure_ascii=False))
    return 0 if all(engineering_seed_gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
