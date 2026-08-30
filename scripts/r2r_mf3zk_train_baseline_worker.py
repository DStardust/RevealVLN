#!/usr/bin/env python3
"""Run one native ETP-R1 R2R-train episode for exact paired baselines.

This worker is deliberately separate from the public-split transfer worker:
its parser accepts only ``train`` and it never constructs a UAD controller.
"""

from __future__ import annotations

import argparse
import json
import math
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


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = base.ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)


def _safe_new(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.exists() or resolved.is_symlink():
        raise RuntimeError(f"run directory must be a new project-local path: {path}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    episode_id = str(args.episode_id)
    run_dir = _safe_new(args.run_dir)
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    base._CONTROLLER = None
    base.MF3B_SCOPE["public_unseen_authorized"] = False
    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3zk_r2r_native_train_{episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.EPISODE_ID", f"['{episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control", "INFERENCE.SPLIT", "train",
        "TASK_CONFIG.DATASET.SUFFIX", "''", "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]", "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-mf3zk-r2r-native-baseline-worker/1",
        "status": "RUNNING", "episode_id": episode_id, "split": "train",
        "mode": "baseline", "revision": "native_etp_r1_baseline",
        "controller": None, "public_unseen_accessed": False,
        "public_unseen_authorized": False,
        "threshold_or_model_tuning_on_r2r": False,
        "future_frames_used": 0, "unseen_or_test_read": False,
        "r2r_checkpoint": str(R2R_CHECKPOINT.relative_to(ROOT)),
        "joint_pretrained": str(JOINT_PRETRAINED.relative_to(ROOT)),
        "argv": argv,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run  # noqa: WPS433

        run.main()
        stats = list(output.rglob("stats_ep_ckpt_270_train_r0_w1.json"))
        if len(stats) != 1:
            raise RuntimeError("native baseline did not produce exactly one train stats file")
        value = json.loads(stats[0].read_text())
        metrics = value.get(episode_id)
        if not isinstance(metrics, dict):
            raise RuntimeError("native baseline stats lack the requested episode")
        summary["metrics"] = {
            key: float(metrics[key]) for key in ("success", "spl", "ndtw", "sdtw")
        }
        if not all(math.isfinite(value) for value in summary["metrics"].values()):
            raise RuntimeError("native baseline metric is non-finite")
        summary["metrics_path"] = str(stats[0].relative_to(ROOT))
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
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        summary["base_trace_sha256"] = base.sha256_file(base_trace)
        summary["paper_result"] = False
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "status": summary["status"], "episode_id": episode_id,
        "metrics": summary.get("metrics"), "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
