#!/usr/bin/env python3
"""Run one frozen MF3ZA episode on the authorized RxR val_unseen split."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "uncertainty", "ensemble"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and project-local")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    base.MF3B_SCOPE["public_unseen_authorized"] = True
    base._CONTROLLER = None
    if args.mode == "uncertainty":
        base._CONTROLLER = base.UncertaintyOnlyController(
            controller_trace, base.MF3ZA_GATE
        )
    elif args.mode == "ensemble":
        base._CONTROLLER = base.MF3KTop2Controller(
            torch.device("cuda:0"), controller_trace, revision="mf3za"
        )
    if base._CONTROLLER is not None:
        base.install_hooks()

    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    sys.argv = [
        "run.py", "--exp_name", f"mf3za_{args.mode}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", "val_unseen", "TASK_CONFIG.DATASET.SPLIT", "val_unseen",
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(base.RXR_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(base.JOINT_PRETRAINED),
        "IL.back_algo", "control", "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "INFERENCE.SPLIT", "val_unseen", "TASK_CONFIG.DATASET.SUFFIX", "''",
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
        "schema_version": "revealnav-mf3za-rxr-unseen-worker/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "mode": args.mode,
        "revision": "mf3za",
        "split": "val_unseen",
        "public_unseen_accessed": True,
        "public_unseen_authorized": True,
        "threshold_tuned_on_val_unseen": False,
        "checkpoint": (
            None if base._CONTROLLER is None else base._CONTROLLER.checkpoint
        ),
        "method_scope": "frozen_mf3za_independent_unseen_evaluation",
    }
    started = time.monotonic()
    try:
        import run

        run.main()
        if base._CONTROLLER is not None:
            actions = [
                json.loads(line)
                for line in base_trace.read_text().splitlines()
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
                    row["authorized"] for row in base._CONTROLLER.records
                ),
                "actions_changed": sum(
                    row["action_changed"] for row in base._CONTROLLER.records
                ),
                "parameters": base._CONTROLLER.parameters,
                "final_record_hash": base._CONTROLLER.previous_hash,
            }
        )
        stats = list(output.rglob("stats_ep_ckpt_1320_val_unseen_r0_w1.json"))
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
