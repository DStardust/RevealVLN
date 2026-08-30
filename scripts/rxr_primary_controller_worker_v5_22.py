#!/usr/bin/env python3
"""Run one paired RxR val_seen episode with the V5.22 controller."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
RXR_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
Q_RESULT = ROOT / (
    "artifacts/evaluation/mf2_branch_excursion_q_v5_1/"
    "RXR_BRANCH_EXCURSION_Q_COMPARISON_V5_1.json"
)
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_full_opp_worker_v5_6 as v56  # noqa: E402
import r2r_remaining_set_rerank_worker_v5_17 as v517  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims,
    sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
V55 = v517.V55
install_runtime_shims()


def expanded_q_checkpoint(seed: int) -> dict:
    result = json.loads(Q_RESULT.read_text())
    if (
        result.get("status") != "BRANCH_EXCURSION_Q_OFFLINE_GATE_PASS"
        or result.get("passing_variants") != ["source_balanced"]
    ):
        raise RuntimeError("expanded RxR Q gate is not frozen-pass")
    row = result["variants"]["source_balanced"]["runs"][str(seed)]
    checkpoint = row["checkpoint"]
    path = (ROOT / checkpoint["path"]).resolve()
    if (
        ROOT not in path.parents
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != checkpoint["bytes"]
        or sha256_file(path) != checkpoint["sha256"]
    ):
        raise RuntimeError("expanded RxR Q checkpoint provenance drift")
    return {**checkpoint, "seed": seed, "variant": "source_balanced"}


def install_expanded_q(state, seed: int) -> dict:
    checkpoint = expanded_q_checkpoint(seed)
    payload = torch.load(
        ROOT / checkpoint["path"], map_location="cpu", weights_only=True
    )
    if (
        payload.get("schema_version")
        != "revealnav-mf2-branch-excursion-q-checkpoint/5.1"
        or payload.get("seed") != seed
        or payload.get("variant") != "source_balanced"
    ):
        raise RuntimeError("expanded RxR Q checkpoint schema drift")
    state.q_model.load_state_dict(payload["model_state_dict"], strict=True)
    state.q_model.to(state.device).eval()
    state.pair = copy.deepcopy(state.pair)
    state.pair["q"] = checkpoint
    return checkpoint


def rxr_base_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "revealnav"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("val_seen",), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if (args.mode == "revealnav") != (args.seed in SEEDS):
        raise SystemExit("revealnav requires one locked seed; baseline forbids seed")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    state = None
    q_checkpoint = None
    if args.mode == "revealnav":
        state = v56.FullOPPActionController(
            args.seed, "revealnav", torch.device("cuda:0"), controller_trace
        )
        q_checkpoint = install_expanded_q(state, args.seed)
        V55._CONTROLLER = state
        V55.install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"rxr_v5_22_{args.mode}_{args.seed}_{args.episode_id}"
    argv = [
        "run.py", "--exp_name", name,
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", args.split,
        "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(RXR_CHECKPOINT),
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "INFERENCE.SPLIT", args.split,
        "TASK_CONFIG.DATASET.SUFFIX", "''",
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
        "schema_version": "revealnav-rxr-primary-worker/5.22",
        "status": "RUNNING",
        "dataset": "RxR-CE-en",
        "episode_id": args.episode_id,
        "seed": args.seed,
        "mode": args.mode,
        "split": args.split,
        "expanded_q_checkpoint": q_checkpoint,
        "val_unseen_or_test_accessed": False,
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        if state is not None:
            state.finalize_episode()
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if state is not None:
            state.finalize_episode()
        pilot.close_envs()
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        summary["base_trace_sha256"] = sha256_file(base_trace)
        summary["controller_trace_sha256"] = (
            sha256_file(controller_trace) if controller_trace.is_file() else None
        )
        summary["controller"] = None if state is None else {
            "strict_load": True,
            "checkpointed_excursions": state.checkpointed_excursions,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
            "final_record_hash": state.previous_hash,
            "checkpoint_triplet": {
                "ree": state.pair["ree"],
                "q": state.pair["q"],
                "post": state.post_row,
            },
        }
        stats = list(output.rglob(
            f"stats_ep_ckpt_1320_{args.split}_r0_w1.json"
        ))
        summary["metrics"] = None
        if len(stats) == 1:
            payload = json.loads(stats[0].read_text())
            summary["metrics"] = payload.get(str(args.episode_id))
            summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        part = run_dir / "RUN_SUMMARY.json.part"
        part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(part, run_dir / "RUN_SUMMARY.json")


def main() -> None:
    mode = None
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--mode":
            mode = sys.argv[index + 1]
        elif value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
    if mode == "baseline":
        rxr_base_main()
        return
    if mode != "revealnav":
        raise SystemExit("--mode baseline|revealnav is required")
    original = v56.main
    v56.main = rxr_base_main
    try:
        v517.main()
    finally:
        v56.main = original
    summary_path = None if run_dir is None else run_dir / "RUN_SUMMARY.json"
    if summary_path is None or not summary_path.is_file():
        raise RuntimeError("V5.22 worker summary is absent")
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-rxr-primary-worker/5.22"
    summary["dataset"] = "RxR-CE-en"
    summary["method_revision"] = "V5.17 executor plus expanded RxR Q V5.1"
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
