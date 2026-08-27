#!/usr/bin/env python3
"""Validate and summarize the fixed three-seed MF2 head-training run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
TRAINING = ROOT / "artifacts/training/mf2_multibranch_v2"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
OUTPUT = TRAINING / "RXR_MULTIBRANCH_HEAD_TRAINING_V2_ACCEPTANCE.json"
SEEDS = (20260826, 20260827, 20260828)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    authorization = json.loads(AUTHORIZATION.read_text())
    manifest_ref = authorization.get("training_manifest", {})
    manifest = (ROOT / manifest_ref.get("path", "__missing__")).resolve()
    failures = []
    if not (
        authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and authorization.get("training_authorized") is True
        and ROOT in manifest.parents
        and manifest.is_file()
        and not manifest.is_symlink()
        and sha256_file(manifest) == manifest_ref.get("sha256")
    ):
        failures.append("training authorization or manifest binding failed")

    runs = []
    for seed in SEEDS:
        summary_path = TRAINING / f"seed_{seed}/training_summary.json"
        if not summary_path.is_file() or summary_path.is_symlink():
            failures.append(f"seed {seed}: summary missing or unsafe")
            continue
        summary = json.loads(summary_path.read_text())
        checkpoint = (ROOT / summary.get("checkpoint", "__missing__")).resolve()
        history = summary.get("history", [])
        totals = [row.get("development", {}).get("total") for row in history]
        finite = all(isinstance(value, (int, float)) and math.isfinite(value)
                     for value in totals)
        best = min(totals) if finite and totals else None
        best_epoch = totals.index(best) + 1 if best is not None else None
        gates = {
            "status_complete": summary.get("status") == "TRAINING_COMPLETE",
            "seed_exact": summary.get("seed") == seed,
            "manifest_path_exact": summary.get("manifest")
            == manifest_ref.get("path"),
            "manifest_sha256_exact": summary.get("manifest_sha256")
            == manifest_ref.get("sha256"),
            "example_counts_exact": summary.get("train_examples") == 280
            and summary.get("development_examples") == 68,
            "fixed_configuration_exact": summary.get("epochs") == 20
            and summary.get("feature_dim") == 768
            and summary.get("budget_count") == 4,
            "history_complete_and_finite": len(history) == 20 and finite,
            "best_value_exact": best is not None
            and abs(summary.get("best_development_total", math.inf) - best)
            <= 1e-12,
            "checkpoint_provenance_exact": checkpoint.is_file()
            and not checkpoint.is_symlink()
            and ROOT in checkpoint.parents
            and checkpoint.stat().st_size == summary.get("checkpoint_bytes")
            and sha256_file(checkpoint) == summary.get("checkpoint_sha256"),
            "causal_boundary_exact": summary.get("future_frames_used") == 0,
            "frozen_backbone_not_retrained": summary.get("backbone_loaded") is False,
            "not_a_paper_result": summary.get("paper_result") is False,
        }
        for name, passed in gates.items():
            if not passed:
                failures.append(f"seed {seed}: {name}")
        runs.append({
            "seed": seed,
            "summary_path": str(summary_path.relative_to(ROOT)),
            "summary_sha256": sha256_file(summary_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT))
            if checkpoint.is_file() and ROOT in checkpoint.parents else None,
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "best_development_total": best,
            "best_epoch": best_epoch,
            "last_development_total": totals[-1] if totals else None,
            "gates": gates,
        })

    values = [run["best_development_total"] for run in runs
              if run["best_development_total"] is not None]
    result = {
        "schema_version": "revealnav-mf2-head-training-acceptance/2",
        "status": "THREE_SEED_TRAINING_PASS_EVALUATION_REQUIRED"
        if not failures and len(runs) == len(SEEDS) else "TRAINING_FAIL",
        "scope": (
            "exploratory supervised causal-head training; effectiveness, "
            "navigation improvement, and paper claims require held-out evaluation"
        ),
        "authorization_path": str(AUTHORIZATION.relative_to(ROOT)),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "training_manifest": manifest_ref,
        "runs": runs,
        "aggregate": {
            "seeds": len(values),
            "best_development_total_mean": statistics.mean(values)
            if values else None,
            "best_development_total_population_std": statistics.pstdev(values)
            if values else None,
            "best_epochs": [run["best_epoch"] for run in runs],
            "overfit_after_early_best_observed": all(
                run["last_development_total"] > run["best_development_total"]
                for run in runs
            ) if runs else False,
        },
        "failures": failures,
        "gold_evaluated": False,
        "baseline_comparison_complete": False,
        "paper_result": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "aggregate": result["aggregate"],
        "failures": failures,
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2))
    return 0 if result["status"].startswith("THREE_SEED_TRAINING_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
