#!/usr/bin/env python3
"""Run one held-out R2R-train confirmation episode.

Modes are native baseline, frozen MF3ZG, and MF3ZK.  MF3ZK reuses the
frozen MF3ZG proposal hierarchy and replaces only its two action-aligned
return gates with the joint train-only gates.  No public split is accepted.
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
from revealnav_mf3.action_aligned import ActionAlignedReturnGate  # noqa: E402


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = base.ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)
GATE_ROOT = ROOT / "artifacts/training/mf3zk_joint_v1/gates"
CONFIRM_PROTOCOL = ROOT / (
    "artifacts/training/mf3zk_joint_v1/"
    "MF3ZK_TRAIN_CONFIRMATION_PROTOCOL.json"
)


def _safe_new(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.exists() or resolved.is_symlink():
        raise RuntimeError(f"run directory must be a new project-local path: {path}")
    return resolved


def _load_joint_gate(tier: str) -> tuple[ActionAlignedReturnGate, dict]:
    gate_path = GATE_ROOT / f"MF3ZK_{tier.upper()}_JOINT_GATE.json"
    if gate_path.is_symlink() or not gate_path.is_file():
        raise RuntimeError(f"missing MF3ZK {tier} gate")
    gate = json.loads(gate_path.read_text())
    if not (
        gate.get("status") == "TRAIN_RETURN_GATE_PASS"
        and gate.get("arm") == "joint"
        and gate.get("tier") == tier
        and gate.get("task_metric_run_authorized") is False
        and gate.get("public_unseen_authorized") is False
        and gate.get("controls", {}).get("unseen_or_test_read") is False
    ):
        raise RuntimeError(f"MF3ZK {tier} gate semantics do not authorize confirmation")
    evidence = gate.get("model", {})
    model_path = (ROOT / str(evidence.get("path"))).resolve()
    if not (
        ROOT in model_path.parents and not model_path.is_symlink()
        and model_path.is_file()
        and int(evidence.get("bytes", -1)) == model_path.stat().st_size
        and evidence.get("sha256") == base.sha256_file(model_path)
    ):
        raise RuntimeError(f"MF3ZK {tier} model provenance drift")
    return ActionAlignedReturnGate(model_path, gate["selected_rule"]), {
        "gate": str(gate_path.relative_to(ROOT)),
        "gate_sha256": base.sha256_file(gate_path),
        "model": str(model_path.relative_to(ROOT)),
        "model_sha256": base.sha256_file(model_path),
    }


def _attach_joint_gates(controller: object) -> dict:
    core, core_evidence = _load_joint_gate("core")
    expansion, expansion_evidence = _load_joint_gate("expansion")
    # MF3ZG has already validated the proposal hierarchy and its frozen
    # backbone.  Only the return/harm screens are replaced here.
    controller.core_return_gate = core
    controller.expansion_return_gate = expansion
    return {"core": core_evidence, "expansion": expansion_evidence}


def _run_stats(output: Path, episode_id: str) -> tuple[Path, dict]:
    stats = list(output.rglob("stats_ep_ckpt_270_train_r0_w1.json"))
    if len(stats) != 1:
        raise RuntimeError("confirmation run did not produce exactly one train stats file")
    payload = json.loads(stats[0].read_text())
    metrics = payload.get(episode_id)
    if not isinstance(metrics, dict):
        raise RuntimeError("confirmation stats lack the requested episode")
    result = {key: float(metrics[key]) for key in ("success", "spl", "ndtw", "sdtw")}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("confirmation metric is non-finite")
    return stats[0], result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "mf3zg", "mf3zk"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    episode_id = str(args.episode_id)
    run_dir = _safe_new(args.run_dir)
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    controller_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    base._CONTROLLER = None
    base.MF3B_SCOPE["public_unseen_authorized"] = False
    gate_evidence = None
    if args.mode == "mf3zg":
        base._CONTROLLER = base.MF3KTop2Controller(
            torch.device("cuda:0"), controller_trace, revision="mf3zg"
        )
        base.install_hooks()
    elif args.mode == "mf3zk":
        base._CONTROLLER = base.MF3KTop2Controller(
            torch.device("cuda:0"), controller_trace, revision="mf3zg"
        )
        gate_evidence = _attach_joint_gates(base._CONTROLLER)
        base.install_hooks()
    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3zk_r2r_confirmation_{args.mode}_{episode_id}",
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
        "schema_version": "revealnav-mf3zk-r2r-train-confirmation-worker/1",
        "status": "RUNNING", "episode_id": episode_id, "split": "train",
        "mode": args.mode, "revision": "mf3zk" if args.mode == "mf3zk" else args.mode,
        "proposal_revision": "mf3zg" if args.mode != "baseline" else None,
        "confirmation_only": True, "public_unseen_accessed": False,
        "public_unseen_authorized": False, "threshold_or_model_tuning_on_r2r": False,
        "future_frames_used": 0, "unseen_or_test_read": False,
        "r2r_checkpoint": str(R2R_CHECKPOINT.relative_to(ROOT)),
        "joint_pretrained": str(JOINT_PRETRAINED.relative_to(ROOT)),
        "joint_gate_evidence": gate_evidence, "argv": argv,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run  # noqa: WPS433

        run.main()
        stats_path, metrics = _run_stats(output, episode_id)
        summary["metrics"] = metrics
        summary["metrics_path"] = str(stats_path.relative_to(ROOT))
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
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        summary["base_trace_sha256"] = base.sha256_file(base_trace)
        summary["controller_trace_sha256"] = base.sha256_file(controller_trace)
        controller = base._CONTROLLER
        summary["controller"] = None if controller is None else {
            "decisions": len(controller.records),
            "authorized": sum(bool(row["authorized"]) for row in controller.records),
            "actions_changed": sum(bool(row["action_changed"]) for row in controller.records),
            "final_record_hash": controller.previous_hash,
        }
        summary["paper_result"] = False
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "status": summary["status"], "episode_id": episode_id,
        "mode": args.mode, "metrics": summary.get("metrics"),
        "actions_changed": (summary.get("controller") or {}).get("actions_changed", 0),
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
