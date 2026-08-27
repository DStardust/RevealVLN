#!/usr/bin/env python3
"""Run one RxR val_unseen episode with a locked V4 shadow controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
LOCK = ROOT / "locks/RXR_UNSEEN_CHECKPOINT_LOCK_V4_2.json"
RXR_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)
for path in reversed((ROOT, ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionMacroController, BranchExcursionQHead,
    BranchMacroAction,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_runtime_shims() -> None:
    import gym.spaces.discrete as discrete_mod

    original_discrete = discrete_mod.Discrete.__init__
    if not getattr(original_discrete, "_revealvln_zero_shim", False):
        def patched_discrete(self, n, *args, **kwargs):
            return original_discrete(self, 1 if n == 0 else n, *args, **kwargs)
        patched_discrete._revealvln_zero_shim = True
        discrete_mod.Discrete.__init__ = patched_discrete

    import vlnce_baselines.common.environments as environments

    original_step = environments.VLNCEDaggerEnv.step
    if getattr(original_step, "_revealvln_trace", False):
        return
    state = {"count": 0}

    def traced_step(self, action, vis_info=None, *args, **kwargs):
        trace_path = os.environ.get("REVEALVLN_BASE_TRACE")
        record = {
            "i": state["count"],
            "act": int(action["act"]),
            "cur_vp": str(action.get("cur_vp")),
            "tryout": bool(action.get("tryout", False)),
            "back_path_len": len(action.get("back_path") or []),
        }
        if int(action["act"]) == 4:
            record.update(
                front_vp=str(action.get("front_vp")),
                ghost_vp=str(action.get("ghost_vp")),
            )
        else:
            record["stop_vp"] = str(action.get("stop_vp"))
        observations, reward, done, info = original_step(
            self, action, vis_info, *args, **kwargs
        )
        record.update(done=bool(done), reward=float(reward))
        state["count"] += 1
        if trace_path:
            with open(trace_path, "a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return observations, reward, done, info

    traced_step._revealvln_trace = True
    environments.VLNCEDaggerEnv.step = traced_step


# Habitat VectorEnv uses forkserver; the child re-imports this module.
install_runtime_shims()


class V4ShadowController:
    def __init__(self, seed: int, device: torch.device, output: Path) -> None:
        lock = json.loads(LOCK.read_text())
        matches = [row for row in lock["checkpoints"] if row["seed"] == seed]
        if len(matches) != 1:
            raise RuntimeError("seed is absent from the frozen checkpoint lock")
        frozen = matches[0]
        checkpoint = ROOT / frozen["path"]
        if (
            checkpoint.stat().st_size != frozen["bytes"]
            or sha256_file(checkpoint) != frozen["sha256"]
        ):
            raise RuntimeError("locked V4 checkpoint provenance drift")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version")
            != "revealnav-mf2-branch-excursion-q-checkpoint/4"
            or payload.get("seed") != seed
        ):
            raise RuntimeError("locked V4 checkpoint schema drift")
        self.model = BranchExcursionQHead(768, 96, 128.0)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(device).eval()
        self.seed = seed
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = frozen["sha256"]
        self.device = device
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("")
        self.controller = BranchExcursionMacroController(3)
        self.instruction: torch.Tensor | None = None
        self.latest_history: torch.Tensor | None = None
        self.rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.previous_current: set[str] = set()
        self.branch_streaks: dict[str, int] = {}
        self.saved_signatures: set[tuple[str, ...]] = set()
        self.previous_hash = "0" * 64
        self.step = 0
        self.decision_rows = 0
        self.checkpoint_rows = 0

    def record_language(self, embeddings, mask) -> None:
        pooled = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        self.instruction = pooled[0].detach()

    def record_panorama(self, embeddings, mask) -> None:
        pooled = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        self.latest_history = pooled[0].detach()

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ) -> None:
        if self.instruction is None or self.latest_history is None:
            return
        current: dict[str, torch.Tensor] = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            current[str(branch_id)] = gmap_img_fts[0, index].detach()
        current_ids = set(current)
        self.branch_streaks = {
            branch_id: (
                self.branch_streaks.get(branch_id, 0) + 1
                if branch_id in self.previous_current else 1
            )
            for branch_id in current_ids
        }
        self.previous_current = current_ids
        signature = tuple(sorted(current_ids))
        persistent = tuple(sorted(
            branch_id for branch_id in current_ids
            if self.branch_streaks[branch_id] >= self.controller.persistence_k
        ))
        if not signature:
            self.step += 1
            return
        self.rows.append((self.latest_history, current))
        ordered_ids = tuple(dict.fromkeys(
            branch_id
            for _, candidates in self.rows
            for branch_id in candidates
        ))
        branch_index = {
            branch_id: index for index, branch_id in enumerate(ordered_ids)
        }
        steps = len(self.rows)
        history = torch.stack([row[0] for row in self.rows]).unsqueeze(0)
        candidates = torch.zeros(
            1, steps, len(ordered_ids), 768, device=self.device
        )
        mask = torch.zeros(
            1, steps, len(ordered_ids), dtype=torch.bool, device=self.device
        )
        for time_index, (_, values) in enumerate(self.rows):
            for branch_id, value in values.items():
                index = branch_index[branch_id]
                candidates[0, time_index, index] = value
                mask[0, time_index, index] = True
        with torch.no_grad():
            result = self.model(
                history, candidates, mask, self.instruction.unsqueeze(0),
                torch.tensor([steps - 1], device=self.device),
            )
        current_indices = [branch_index[branch_id] for branch_id in persistent]
        commit_costs = [
            float(result.commit_cost[0, index]) for index in current_indices
        ]
        excursion_costs = [
            float(result.excursion_cost[0, index]) for index in current_indices
        ]
        decision = self.controller.decide(
            persistent, commit_costs, excursion_costs,
            self.controller.persistence_k,
        )
        eligible = decision.action is not BranchMacroAction.DEFER
        checkpoint_created = (
            eligible
            and decision.action is BranchMacroAction.CHECKPOINTED_EXCURSION
            and persistent not in self.saved_signatures
        )
        if eligible:
            self.decision_rows += 1
        if checkpoint_created:
            self.saved_signatures.add(persistent)
            self.checkpoint_rows += 1
        row = {
            "step": self.step,
            "candidate_count": len(signature),
            "persistent_candidate_count": len(persistent),
            "minimum_persistent_streak": (
                min(self.branch_streaks[branch_id] for branch_id in persistent)
                if persistent else 0
            ),
            "decision_eligible": eligible,
            "macro_action": decision.action.value,
            "branch_id": decision.branch_id,
            "predicted_cost": None if decision.predicted_cost is None else round(
                decision.predicted_cost, 8
            ),
            "preservation_gain": (
                None if decision.preservation_gain is None else round(
                    decision.preservation_gain, 8
                )
            ),
            "checkpoint_created": checkpoint_created,
            "shadow_only_not_executed": True,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["record_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        self.previous_hash = row["record_hash"]
        with self.output.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1


_CONTROLLER: V4ShadowController | None = None


def controller() -> V4ShadowController | None:
    global _CONTROLLER
    if str(os.getpid()) != os.environ.get("REVEALVLN_CONTROLLER_MAIN_PID"):
        return None
    output = os.environ.get("REVEALVLN_CONTROLLER_TRACE")
    seed = os.environ.get("REVEALVLN_CONTROLLER_SEED")
    if not output or not seed:
        return None
    if _CONTROLLER is None:
        _CONTROLLER = V4ShadowController(
            int(seed), torch.device("cuda:0"), Path(output)
        )
    return _CONTROLLER


def install_forward_controller() -> None:
    from vlnce_baselines.models.R1Policy import ETP

    original_forward = ETP.forward
    if getattr(original_forward, "_revealvln_v4_shadow", False):
        return

    def wrapped(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else None)
        state = controller()
        if state is None:
            return result
        if mode == "language":
            state.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            state.record_panorama(result[0], result[1])
        elif mode == "navigation":
            state.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
            )
        return result

    wrapped._revealvln_v4_shadow = True
    ETP.forward = wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be a new directory inside the project")
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "base_trace.jsonl"
    controller_path = run_dir / "v4_controller.jsonl"
    trace_path.write_text("")
    os.environ.update({
        "REVEALVLN_BASE_TRACE": str(trace_path),
        "REVEALVLN_CONTROLLER_MAIN_PID": str(os.getpid()),
        "REVEALVLN_CONTROLLER_TRACE": str(controller_path),
        "REVEALVLN_CONTROLLER_SEED": str(args.seed),
    })
    install_forward_controller()
    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output_root = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", args.exp_name,
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", "val_unseen",
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(RXR_CHECKPOINT),
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output_root / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output_root / "checkpoints"),
        "RESULTS_DIR", str(output_root / "results"),
    ]
    summary = {
        "schema_version": "revealnav-rxr-unseen-controller-worker/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "seed": args.seed,
        "split": "val_unseen",
        "languages": ["en-US", "en-IN"],
        "controller_mode": "locked_v4_shadow_only",
        "shadow_actions_executed": 0,
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
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
        state = controller()
        summary["controller"] = None if state is None else {
            "checkpoint_path": str(state.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": state.checkpoint_sha256,
            "strict_load": True,
            "rows": len(state.rows),
            "decision_rows": state.decision_rows,
            "checkpoint_rows": state.checkpoint_rows,
            "final_record_hash": state.previous_hash,
        }
        summary["base_trace_sha256"] = sha256_file(trace_path)
        summary["controller_trace_sha256"] = (
            sha256_file(controller_path) if controller_path.is_file() else None
        )
        stats = list(output_root.rglob(
            "stats_ep_ckpt_1320_val_unseen_r0_w1.json"
        ))
        summary["metrics"] = None
        if len(stats) == 1:
            payload = json.loads(stats[0].read_text())
            summary["metrics"] = payload.get(str(args.episode_id))
            summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
