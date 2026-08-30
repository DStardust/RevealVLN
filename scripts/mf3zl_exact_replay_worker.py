#!/usr/bin/env python3
"""One project-local MF3ZL native-shadow or targeted-switch rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (SCRIPTS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import rxr_uad_controller_worker_mf3 as base  # noqa: E402
from revealnav_mf3.exact_replay import (  # noqa: E402
    ExactReplayController,
    ProposalEventIdentity,
    validate_forced_switch,
    validate_shadow_event,
)


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
RXR_CHECKPOINT = base.RXR_CHECKPOINT
JOINT_PRETRAINED = base.JOINT_PRETRAINED


def _new_project_path(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.exists() or resolved.is_symlink():
        raise RuntimeError(f"MF3ZL run path must be new and project-local: {path}")
    return resolved


def _target(path: Path | None) -> ProposalEventIdentity | None:
    if path is None:
        return None
    resolved = path.resolve()
    if ROOT not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError("MF3ZL target must be a regular project-local file")
    value = json.loads(resolved.read_text())
    return ProposalEventIdentity(**value["event_identity"])


def _argv(dataset: str, episode_id: str, output: Path) -> list[str]:
    if dataset == "RxR":
        config = "run_rxr/iter_train.yaml"
        checkpoint = RXR_CHECKPOINT
        extra = [
            "EVAL.LANGUAGES", "['en-US','en-IN']",
            "IL.RECOLLECT_TRAINER.gt_file",
            "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        ]
    elif dataset == "R2R":
        config = "run_r2r/iter_train.yaml"
        checkpoint = R2R_CHECKPOINT
        extra = []
    else:
        raise ValueError("unsupported exact-replay dataset")
    return [
        "run.py", "--exp_name", f"mf3zl_{dataset.lower()}_{episode_id}",
        "--run-type", "eval", "--exp-config", config,
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        *extra,
        "EVAL.EPISODE_ID", f"['{episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(checkpoint), "EVAL.SAMPLE", "False",
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


def _stats_inventory(output: Path, dataset: str) -> list[dict]:
    checkpoint = "1320" if dataset == "RxR" else "270"
    matches = list(output.rglob(f"stats_ep_ckpt_{checkpoint}_train_r0_w1.json"))
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
            "payload_read_by_worker": False,
        }
        for path in matches
    ]


def _event_inventory(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        path = Path(event["feature_path"]).resolve()
        if ROOT not in path.parents or not path.is_file() or path.is_symlink():
            raise RuntimeError("MF3ZL event feature is not a regular project-local file")
        result.append({
            "event_identity": event["event_identity"],
            "decision": event["decision"],
            "feature": {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            },
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("RxR", "R2R"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--mode", choices=("native_shadow", "targeted_switch"), required=True
    )
    parser.add_argument("--target", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    target = _target(args.target)
    if (args.mode == "native_shadow") != (target is None):
        raise RuntimeError("MF3ZL worker target/mode mismatch")
    run_dir = _new_project_path(args.run_dir)
    run_dir.mkdir(parents=True)
    feature_dir = run_dir / "features"
    feature_dir.mkdir()
    base_trace = run_dir / "base_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    proposal_trace = run_dir / "proposal_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    proposal = base.MF3KTop2Controller(
        torch.device("cuda:0"), proposal_trace, revision="mf3zg"
    )
    controller = ExactReplayController(
        proposal, controller_trace, feature_dir,
        dataset=args.dataset, episode_id=str(args.episode_id),
        scene_id=args.scene_id, mode=args.mode, target=target,
    )
    base._CONTROLLER = controller
    base.install_hooks()
    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = _argv(args.dataset, str(args.episode_id), output)
    summary = {
        "schema_version": "revealnav-mf3zl-exact-replay-worker/1",
        "status": "RUNNING",
        "dataset": args.dataset,
        "split": "train",
        "episode_id": str(args.episode_id),
        "scene_id": args.scene_id,
        "mode": args.mode,
        "target": None if target is None else asdict(target),
        "proposal_revision": "mf3zg",
        "public_split_access": False,
        "task_metric_payload_read": False,
        "future_observation_used": False,
        "argv": argv,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run  # noqa: WPS433

        run.main()
        actions = [
            json.loads(line) for line in base_trace.read_text().splitlines() if line
        ]
        summary["executed_action_validation"] = base.verify_execution(
            controller.records, actions
        )
        if args.mode == "native_shadow":
            for record in controller.records:
                if record["event_identity"] is not None:
                    validate_shadow_event(record)
            if controller.switched or any(row["action_changed"] for row in controller.records):
                raise RuntimeError("native shadow changed an action")
        else:
            if not controller.switched:
                raise RuntimeError("sealed targeted switch was never executed")
            validate_forced_switch(controller.records, target)
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "decisions": len(controller.records),
            "proposal_events": _event_inventory(controller.events),
            "changed_actions": sum(bool(row["action_changed"]) for row in controller.records),
            "final_record_hash": controller.previous_hash,
            "checkpoint_inventory": controller.checkpoint,
            "base_trace": {
                "path": str(base_trace.relative_to(ROOT)),
                "bytes": base_trace.stat().st_size,
                "sha256": base.sha256_file(base_trace),
            },
            "controller_trace": {
                "path": str(controller_trace.relative_to(ROOT)),
                "bytes": controller_trace.stat().st_size,
                "sha256": base.sha256_file(controller_trace),
            },
            "proposal_trace": {
                "path": str(proposal_trace.relative_to(ROOT)),
                "bytes": proposal_trace.stat().st_size,
                "sha256": base.sha256_file(proposal_trace),
            },
            "stats_inventory": _stats_inventory(output, args.dataset),
            "paper_result": False,
        })
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "status": summary["status"],
        "dataset": args.dataset,
        "episode_id": str(args.episode_id),
        "mode": args.mode,
        "proposal_events": len(controller.events),
        "changed_actions": summary["changed_actions"],
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
