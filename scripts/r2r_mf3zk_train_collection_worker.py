#!/usr/bin/env python3
"""Collect exact one-switch MF3V counterfactuals on R2R train only.

This worker is intentionally separate from the frozen MF3ZG transfer worker.
It uses the already sealed MF3ZF proposal band in collection-only mode, so it
never reads a public evaluation split and never loads a learned return gate.
"""

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

# The base controller checks this flag while constructing MF3ZF.  It must be
# set before import so forked/re-imported modules observe the same mode.
os.environ["REVEALNAV_MF3ZF_COLLECTION_ONLY"] = "1"
import rxr_uad_controller_worker_mf3 as base  # noqa: E402


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = base.ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)
COLLECTION_GATE = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)


def _safe_new(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.exists() or resolved.is_symlink():
        raise RuntimeError(f"run output must be a new project-local path: {path}")
    return resolved


def _stats(output: Path, episode_id: str) -> tuple[Path, dict] | None:
    matches = list(output.rglob("stats_ep_ckpt_270_train_r0_w1.json"))
    if len(matches) != 1:
        return None
    value = json.loads(matches[0].read_text())
    metrics = value.get(str(episode_id))
    if not isinstance(metrics, dict):
        return None
    return matches[0], metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    episode_id = str(args.episode_id)
    run_dir = _safe_new(args.run_dir)
    if not (
        COLLECTION_GATE.is_file() and not COLLECTION_GATE.is_symlink()
        and ROOT in COLLECTION_GATE.resolve().parents
    ):
        raise RuntimeError("MF3ZF collection gate is unavailable")
    gate = json.loads(COLLECTION_GATE.read_text())
    if not (
        gate.get("status") == "TRAIN_RETURN_COLLECTION_AUTHORIZED"
        and gate.get("task_metric_run_authorized") is False
        and gate.get("collection_split") == "train"
        and gate.get("unseen_or_test_read") is False
    ):
        raise RuntimeError("MF3ZF collection gate semantics drift")

    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    feature_path = run_dir / "intervention_feature.npz"
    base_trace.write_text("")
    controller_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    os.environ["REVEALNAV_MF3_INTERVENTION_FEATURE"] = str(feature_path)

    base._CONTROLLER = base.MF3KTop2Controller(
        torch.device("cuda:0"), controller_trace, revision="mf3zf"
    )
    base.install_hooks()
    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3zk_r2r_collection_{episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.EPISODE_ID", f"['{episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", "train", "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]", "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-mf3zk-r2r-collection-worker/1",
        "status": "RUNNING", "episode_id": episode_id, "split": "train",
        "mode": "exact_mf3v_collection", "revision": "mf3zk",
        "proposal_revision": "mf3zf", "collection_only": True,
        "task_metric_payload_read": False,
        "ground_truth_payload_read": False,
        "future_frames_used": 0, "unseen_or_test_read": False,
        "native_action_overridden": False,
        "collection_gate_sha256": base.sha256_file(COLLECTION_GATE),
        "argv": argv,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run  # noqa: WPS433

        run.main()
        actions = [
            json.loads(line) for line in base_trace.read_text().splitlines()
            if line
        ]
        summary["executed_action_validation"] = base.verify_execution(
            base._CONTROLLER.records, actions
        )
        changed = [
            row for row in base._CONTROLLER.records
            if row.get("action_changed") is True
        ]
        summary["status"] = "PASS"
        summary["changed_actions"] = len(changed)
        summary["intervention_feature_written"] = feature_path.is_file()
        if len(changed) > 1:
            raise RuntimeError("collection worker changed more than one action")
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state = base._CONTROLLER
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
            "controller": None if state is None else {
                "decisions": len(state.records),
                "authorized": sum(bool(row["authorized"]) for row in state.records),
                "actions_changed": sum(
                    bool(row["action_changed"]) for row in state.records
                ),
                "parameters": state.parameters,
                "final_record_hash": state.previous_hash,
            },
            "feature": (
                {"path": str(feature_path.relative_to(ROOT)),
                 "bytes": feature_path.stat().st_size,
                 "sha256": base.sha256_file(feature_path)}
                if feature_path.is_file() else None
            ),
            "base_trace_sha256": base.sha256_file(base_trace),
            "controller_trace_sha256": base.sha256_file(controller_trace),
            "paper_result": False,
        })
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "status": summary["status"], "episode_id": episode_id,
        "changed_actions": summary.get("changed_actions"),
        "feature_written": summary.get("intervention_feature_written"),
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
