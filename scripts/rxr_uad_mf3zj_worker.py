#!/usr/bin/env python3
"""Run one RxR episode with MF3ZJ's globally budgeted controller."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import rxr_uad_controller_worker_mf3 as base  # noqa: E402
from rxr_uad_mf3zj_controller import MF3ZJController  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "uncertainty", "ensemble"),
        default="ensemble",
    )
    parser.add_argument(
        "--split", choices=("train", "val_seen", "val_unseen"),
        default="val_unseen",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("MF3ZJ run directory must be new and project-local")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    base.MF3B_SCOPE["public_unseen_authorized"] = args.split == "val_unseen"
    base._CONTROLLER = None
    if args.mode == "ensemble":
        base._CONTROLLER = MF3ZJController(
            torch.device("cuda:0"), controller_trace, run_dir
        )
    elif args.mode == "uncertainty":
        base._CONTROLLER = base.UncertaintyOnlyController(
            controller_trace, base.MF3V_GATE
        )
    if base._CONTROLLER is not None:
        base.install_hooks()

    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3zj_{args.mode}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(base.RXR_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(base.JOINT_PRETRAINED),
        "IL.back_algo", "control", "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "INFERENCE.SPLIT", args.split, "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-mf3zj-rxr-worker/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "mode": args.mode,
        "revision": "mf3zj",
        "split": args.split,
        "public_unseen_accessed": args.split == "val_unseen",
        "current_holdout_used_for_tuning": False,
        "threshold_tuned_on_val_unseen": False,
        "prior_val_unseen_used_for_failure_analysis": True,
        "test_or_test_challenge_accessed": False,
        "checkpoint": (
            None if base._CONTROLLER is None else base._CONTROLLER.checkpoint
        ),
        "method_scope": "uad_counterfactual_transfer_arbitration",
        **base.MF3B_SCOPE,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run  # noqa: WPS433

        run.main()
        if base._CONTROLLER is not None:
            actions = [
                json.loads(line) for line in base_trace.read_text().splitlines()
                if line
            ]
            summary["executed_action_validation"] = base.verify_execution(
                base._CONTROLLER.records, actions
            )
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["controller"] = (
            None if base._CONTROLLER is None else {
                "decisions": len(base._CONTROLLER.records),
                "authorized": sum(
                    bool(row["authorized"]) for row in base._CONTROLLER.records
                ),
                "actions_changed": sum(
                    bool(row["action_changed"])
                    for row in base._CONTROLLER.records
                ),
                "decision_sources": {
                    source: sum(
                        row.get("decision_source") == source
                        for row in base._CONTROLLER.records
                    )
                    for source in (
                        "learned_residual", "counterfactual_fallback"
                    )
                },
                "parameters": base._CONTROLLER.parameters,
                "final_record_hash": base._CONTROLLER.previous_hash,
            }
        )
        stats = list(output.rglob(
            f"stats_ep_ckpt_1320_{args.split}_r0_w1.json"
        ))
        summary["metrics"] = None
        if len(stats) == 1:
            summary["metrics"] = json.loads(stats[0].read_text()).get(
                str(args.episode_id)
            )
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
