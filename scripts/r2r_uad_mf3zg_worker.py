#!/usr/bin/env python3
"""Run one R2R episode with the RxR-frozen MF3ZG controller."""

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


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = base.ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)


def _validate_unseen_authorization(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("R2R unseen authorization must be a project-local file")
    expected = os.environ.get("REVEALNAV_R2R_UNSEEN_PROTOCOL_SHA256")
    if not expected or base.sha256_file(resolved) != expected:
        raise RuntimeError("R2R unseen authorization digest mismatch")
    value = json.loads(resolved.read_text())
    if not (
        value.get("status") == "SEALED_BEFORE_R2R_VAL_UNSEEN_METRICS"
        and value.get("controller_revision") == "mf3zg"
        and value.get("threshold_or_model_tuning_on_r2r") is False
    ):
        raise RuntimeError("R2R unseen authorization semantics mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "ensemble"), required=True)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization-json", type=Path)
    args = parser.parse_args()
    if (args.split == "val_unseen") != (args.authorization_json is not None):
        raise SystemExit("val_unseen requires its sealed authorization; val_seen forbids it")
    authorization = _validate_unseen_authorization(args.authorization_json)

    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and project-local")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    base._CONTROLLER = None
    base.MF3B_SCOPE["public_unseen_authorized"] = args.split == "val_unseen"
    if args.mode == "ensemble":
        base._CONTROLLER = base.MF3KTop2Controller(
            torch.device("cuda:0"), controller_trace, revision="mf3zg"
        )
        base.install_hooks()

    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3zg_r2r_{args.mode}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control", "INFERENCE.SPLIT", args.split,
        "TASK_CONFIG.DATASET.SUFFIX", "''", "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-mf3zg-r2r-worker/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "mode": args.mode,
        "revision": "mf3zg",
        "split": args.split,
        **base.MF3B_SCOPE,
        "public_unseen_accessed": args.split == "val_unseen",
        "public_unseen_authorized": authorization is not None,
        "threshold_or_model_tuning_on_r2r": False,
        "parameters_frozen_from_rxr": True,
        "r2r_checkpoint": str(R2R_CHECKPOINT.relative_to(ROOT)),
        "controller_checkpoints": (
            None if base._CONTROLLER is None else base._CONTROLLER.checkpoint
        ),
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
        state = base._CONTROLLER
        summary["controller"] = None if state is None else {
            "decisions": len(state.records),
            "authorized": sum(bool(row["authorized"]) for row in state.records),
            "actions_changed": sum(
                bool(row["action_changed"]) for row in state.records
            ),
            "parameters": state.parameters,
            "final_record_hash": state.previous_hash,
        }
        stats = list(output.rglob(
            f"stats_ep_ckpt_270_{args.split}_r0_w1.json"
        ))
        summary["metrics"] = None
        if len(stats) == 1:
            summary["metrics"] = json.loads(stats[0].read_text()).get(
                str(args.episode_id)
            )
            summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
