#!/usr/bin/env python3
"""Export a compact, reviewable view of the full MF3ZM-CAR result."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "artifacts/training/mf3zm_car_v1/"
    "MF3ZM_CAR_TRAIN_DEVELOPMENT_RESULT.json"
)
OUTPUT = ROOT / "MF3ZM_CAR_REVIEW_RESULT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(value: dict) -> dict:
    names = (
        "authorized", "total_utility", "selected_utility",
        "catastrophic_rate", "minimum_leave_one_selected_scene_out_total",
    )
    return {name: value.get(name) for name in names if name in value}


def reason_category(reason: str) -> str:
    suffixes = (
        "utility_not_above_low_native_margin",
        "utility_not_above_high_proposal_score",
        "catastrophic_rate_above_low_native_margin",
        "catastrophic_rate_above_high_proposal_score",
        "negative_utility", "zero_intervention",
    )
    for suffix in suffixes:
        if reason.endswith(suffix):
            return suffix
    return reason


def slim_baselines(value: dict) -> dict:
    source = value.get("fold_domain_matched", {}).get("baselines", {})
    result = {}
    for name in ("low_native_margin", "high_proposal_score",
                 "fixed_seed_random"):
        if name not in source:
            continue
        result[name] = {
            "overall": metrics(source[name].get("overall", {})),
            "domains": {
                domain: metrics(item)
                for domain, item in source[name].get("domains", {}).items()
            },
        }
    return result


def slim_trial(value: dict) -> dict:
    evidence = value.get("evidence", {})
    return {
        "weight_decay": value.get("weight_decay"),
        "feasible": value.get("feasible"),
        "selection_loss": value.get(
            "inner_oof_preference_loss",
            value.get("inner_oof_quantile_loss"),
        ),
        "failure_reason_counts": dict(sorted(Counter(
            reason_category(reason)
            for reason in value.get("failure_reasons", [])
        ).items())),
        "overall": metrics(evidence.get("overall", {})),
        "domains": {
            domain: metrics(item)
            for domain, item in evidence.get("domains", {}).items()
        },
        "fold_domain_matched_baselines": slim_baselines(
            value.get("equal_budget", {})
        ),
    }


def slim_arm(value: dict) -> dict:
    return {
        "status": value.get("status"),
        "failure_reasons": value.get("failure_reasons", []),
        "outer_folds": [
            {
                "outer_fold": fold.get("outer_fold"),
                "selected_weight_decay": fold.get("selected_weight_decay"),
                "trials": [slim_trial(trial)
                           for trial in fold.get("trials", [])],
            }
            for fold in value.get("outer_folds", [])
        ],
    }


def main() -> int:
    source = json.loads(SOURCE.read_text())
    result = {
        "schema_version": "revealnav-mf3zm-car-review-result/1",
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256_file(SOURCE),
        },
        "status": source.get("status"),
        "rows": source.get("rows"),
        "scenes": source.get("scenes"),
        "domains": source.get("domains"),
        "checkpoint_created": source.get("checkpoint_created"),
        "public_unseen_authorized": source.get("public_unseen_authorized"),
        "execution_mode": source.get("execution_mode"),
        "control_errors": source.get("control_errors"),
        "mainline": slim_arm(source.get("mainline", {})),
        "controls": {
            name: slim_arm(value)
            for name, value in source.get("controls", {}).items()
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT.relative_to(ROOT)),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256_file(OUTPUT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
