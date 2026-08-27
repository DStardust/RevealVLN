#!/usr/bin/env python3
"""Run one R2R episode with the full frozen OPP event gate."""

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
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
R2R_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
CALIBRATION = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/"
    "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
)
for path in reversed((ROOT, ROOT / "scripts", ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_continuous_controller_worker_v5_2 as v52  # noqa: E402
import r2r_continuous_controller_worker_v5_3 as v53  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction, FrozenOPPEventGate, PostExcursionAction,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()


def frozen_config() -> dict:
    payload = json.loads(CALIBRATION.read_text())
    config = payload.get("selected_shared_config", {})
    expected = {
        "active_width": 2,
        "discriminable_threshold": 0.7,
        "evidence_threshold": 0.5,
        "expiry_threshold": 0.3,
        "opv_threshold": 0.025,
        "persistence_k": 3,
        "retrieval_limit": 8,
        "reveal_threshold": 0.5,
        "target_threshold": 0.3,
        "wrong_commitment_weight": 5.0,
    }
    if (
        payload.get("status")
        != "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_PASS"
        or payload.get("gold_payload_read") is not False
        or config != expected
    ):
        raise RuntimeError("frozen shared OPP calibration is invalid")
    return expected


FROZEN_CONFIG = frozen_config()


class RecordingREE(torch.nn.Module):
    """Expose the already-computed frozen REE output to the OPP gate."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.last_output = None

    def forward(self, *args, **kwargs):
        self.last_output = self.model(*args, **kwargs)
        return self.last_output


class FullOPPContinuousController(v53.PersistentContinuousController):
    """V5.3 motion execution constrained by all frozen OPP event heads."""

    def __init__(
        self, seed: int, mode: str, device: torch.device, trace_path: Path,
    ) -> None:
        if mode not in ("shadow", "revealnav"):
            raise ValueError("V5.4 controller mode must be shadow or revealnav")
        super().__init__(seed, device, trace_path)
        self.mode = mode
        self.ree_model = RecordingREE(self.ree_model)
        self.event_gate = FrozenOPPEventGate(
            FROZEN_CONFIG["discriminable_threshold"],
            FROZEN_CONFIG["evidence_threshold"],
            FROZEN_CONFIG["target_threshold"],
            FROZEN_CONFIG["expiry_threshold"],
            FROZEN_CONFIG["reveal_threshold"],
        )
        self.checkpoint_candidates: dict[str, torch.Tensor] = {}
        self.opp_checkpoint_acceptances = 0
        self.opp_checkpoint_suppressions: dict[str, int] = {}
        self.shadow_activations = 0
        self.raw_post_continues = 0
        self.raw_post_backtracks = 0
        self.opp_forced_backtracks = 0
        self.terminal_unresolved_excursions = 0
        self._episode_finalized = False

    def _reset_search(self) -> None:
        super()._reset_search()
        self.checkpoint_candidates = {}

    @staticmethod
    def _probability(value: torch.Tensor) -> float:
        return float(torch.sigmoid(value).detach().cpu())

    def _event_belief(
        self, output, step: int, candidate_indices: list[int],
    ) -> dict[str, float]:
        if output is None or not candidate_indices:
            raise RuntimeError("REE event output or candidate set is absent")
        probabilities = torch.softmax(
            output.target_logits[0, step, candidate_indices], dim=-1
        )
        target_in_set = self._probability(output.target_in_set_logit[0, step])
        separation = self._probability(output.separation_logit[0, step])
        evidence = self._probability(output.evidence_logit[0, step])
        return {
            "p_unobserved": 1.0 - target_in_set,
            "p_ambiguous": target_in_set * (1.0 - separation * evidence),
            "p_discriminable": target_in_set * separation * evidence,
            "evidence": evidence,
            "reveal_hazard": self._probability(
                output.reveal_hazard_logit[0, step]
            ),
            "expiry_hazard": self._probability(
                output.expiry_hazard_logit[0, step]
            ),
            "maximum_target_probability": float(probabilities.max().cpu()),
        }

    def _initial_belief(self, persistent: tuple[str, ...]) -> dict[str, float]:
        ordered = tuple(dict.fromkeys(
            branch_id for _, values in self.rows for branch_id in values
        ))
        index = {branch_id: value for value, branch_id in enumerate(ordered)}
        if any(branch_id not in index for branch_id in persistent):
            raise RuntimeError("persistent branch absent from REE candidate tensor")
        return self._event_belief(
            self.ree_model.last_output, len(self.rows) - 1,
            [index[branch_id] for branch_id in persistent],
        )

    def _initial_decision(self, current, persistent):
        self.fusion.last_decision = None
        selected = pilot.ActionEnabledPilotController._initial_decision(
            self, current, persistent
        )
        decision = self.fusion.last_decision
        if selected is None:
            if (
                decision is not None
                and decision.reason
                == "checkpoint_value_not_above_frozen_opv_threshold"
            ):
                self.threshold_suppressions += 1
            return None
        if (
            decision is None
            or decision.action is not BranchMacroAction.CHECKPOINTED_EXCURSION
            or decision.branch_id != selected
        ):
            raise RuntimeError("REE-Q proposal state is inconsistent")
        belief = self._initial_belief(persistent)
        allowed, reason = self.event_gate.checkpoint_decision(
            belief["p_discriminable"], belief["evidence"],
            belief["maximum_target_probability"], belief["reveal_hazard"],
            belief["expiry_hazard"],
        )
        self.record(
            "opp_checkpoint_gate", allowed=allowed, reason=reason,
            branch_id=selected,
            preservation_gain=round(float(decision.preservation_gain), 8),
            **{key: round(value, 8) for key, value in belief.items()},
        )
        if not allowed:
            self.opp_checkpoint_suppressions[reason] = (
                self.opp_checkpoint_suppressions.get(reason, 0) + 1
            )
            self._reset_search()
            return None
        self.opp_checkpoint_acceptances += 1
        if self.mode == "shadow":
            self.shadow_activations += 1
            self.record(
                "shadow_opp_activation", checkpoint_id=self.checkpoint_id,
                branch_id=selected, shadow_only_not_executed=True,
            )
            self._reset_search()
            return None
        if not self.ledger.authorize(self.checkpoint_id, decision):
            raise RuntimeError("ECOG ledger rejected an OPP-approved excursion")
        self.checkpoint_candidates = {
            branch_id: current[branch_id].detach()
            for branch_id in persistent
        }
        self.record(
            "ecog_opp_authorized", checkpoint_id=self.checkpoint_id,
            branch_id=selected,
            preservation_gain=round(float(decision.preservation_gain), 8),
            opv_threshold=FROZEN_CONFIG["opv_threshold"],
        )
        return selected

    def _post_ree_belief(self, current) -> dict[str, float]:
        histories = [*self.pre_histories, self.latest_history.detach()]
        values = dict(self.checkpoint_candidates)
        values.update(current)
        values[self.selected_branch] = self.selected_embedding
        ordered = tuple(sorted(values))
        index = {branch_id: value for value, branch_id in enumerate(ordered)}
        steps = len(histories)
        history = torch.stack(histories).unsqueeze(0)
        candidates = torch.zeros(
            1, steps, len(ordered), 768, device=self.device
        )
        mask = torch.zeros(
            1, steps, len(ordered), dtype=torch.bool, device=self.device
        )
        for time_index, (_, row_values) in enumerate(self.rows):
            for branch_id, embedding in row_values.items():
                if branch_id in index:
                    candidates[0, time_index, index[branch_id]] = embedding
                    mask[0, time_index, index[branch_id]] = True
        for branch_id, embedding in values.items():
            candidates[0, steps - 1, index[branch_id]] = embedding
            mask[0, steps - 1, index[branch_id]] = True
        budgets = torch.tensor(
            [1.5, 2.0, 3.0, 4.0], device=self.device
        ).view(1, 1, 4).expand(1, steps, 4)
        with torch.no_grad():
            output = self.ree_model(
                history, candidates, mask, budgets,
                self.instruction.unsqueeze(0),
            )
        belief = self._event_belief(
            output, steps - 1, list(range(len(ordered)))
        )
        probabilities = torch.softmax(
            output.target_logits[0, steps - 1], dim=-1
        )
        belief["selected_target_probability"] = float(
            probabilities[index[self.selected_branch]].cpu()
        )
        return belief

    def _post_decision(self, current) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        ).unsqueeze(0)
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        with torch.no_grad():
            output = self.post_model(
                history, torch.tensor([history.shape[1]], device=self.device),
                self.instruction.unsqueeze(0),
                self.selected_embedding.unsqueeze(0),
                self.checkpoint_embedding.unsqueeze(0), local.unsqueeze(0),
                torch.tensor([1.0], device=self.device),
            )
        continue_cost = float(output.continue_cost[0])
        backtrack_cost = float(output.backtrack_cost[0])
        raw_continue = continue_cost <= backtrack_cost
        if raw_continue:
            self.raw_post_continues += 1
        else:
            self.raw_post_backtracks += 1
        belief = self._post_ree_belief(current)
        opp_continue, reason = self.event_gate.post_excursion_decision(
            belief["p_discriminable"], belief["evidence"],
            belief["selected_target_probability"],
        )
        execute_continue = raw_continue and opp_continue
        action = (
            PostExcursionAction.CONTINUE
            if execute_continue else PostExcursionAction.BACKTRACK
        )
        if raw_continue and not opp_continue:
            self.opp_forced_backtracks += 1
        self.post_policy_action = action.value
        self.record(
            "post_decision", policy_action=action.value,
            raw_post_q_action=("continue" if raw_continue else "backtrack"),
            opp_action=("continue" if opp_continue else "backtrack"),
            opp_reason=reason,
            predicted_continue_cost=round(continue_cost, 8),
            predicted_backtrack_cost=round(backtrack_cost, 8),
            executed_return=not execute_continue,
            forced_stress_return=False,
            **{key: round(value, 8) for key, value in belief.items()},
        )
        if execute_continue:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self.ledger.resolve_continue(checkpoint_id, branch_id)
            self._reset_search()
        else:
            self.backtrack_decisions += 1
            self._schedule_return()

    def finalize_episode(self) -> None:
        if self._episode_finalized:
            return
        self._episode_finalized = True
        if self.phase == "outbound_in_flight":
            self.terminal_unresolved_excursions += 1
            self.record(
                "terminal_excursion_without_post_decision",
                checkpoint_id=self.checkpoint_id,
                branch_id=self.selected_branch,
                fail_closed=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "shadow", "revealnav"), required=True
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if (args.mode != "baseline") != (args.seed is not None):
        raise SystemExit("shadow/revealnav require a seed; baseline forbids one")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    state = None
    if args.mode != "baseline":
        state = FullOPPContinuousController(
            args.seed, args.mode, torch.device("cuda:0"), controller_trace
        )
        v52._CONTROLLER = state
        v52.install_continuous_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"continuous_v5_4_{args.mode}_{args.seed}_{args.episode_id}"
    argv = [
        "run.py", "--exp_name", name,
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
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
        "schema_version": "revealnav-r2r-continuous-controller-worker/5.4",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": args.mode, "split": args.split,
        "frozen_opp_config": FROZEN_CONFIG,
        "task_metric_payload_read": args.mode != "shadow", "argv": argv,
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
        if state is not None:
            state.finalize_episode()
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        summary["controller"] = None if state is None else {
            "strict_load": True,
            "checkpointed_excursions": state.checkpointed_excursions,
            "opp_checkpoint_acceptances": state.opp_checkpoint_acceptances,
            "opp_checkpoint_suppressions": state.opp_checkpoint_suppressions,
            "shadow_activations": state.shadow_activations,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "raw_post_continues": state.raw_post_continues,
            "raw_post_backtracks": state.raw_post_backtracks,
            "opp_forced_backtracks": state.opp_forced_backtracks,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
            "terminal_unresolved_excursions": state.terminal_unresolved_excursions,
            "threshold_suppressions": state.threshold_suppressions,
            "ledger_suppressions": state.ledger_suppressions,
            "ledger_counts": state.ledger.counts(),
            "final_record_hash": state.previous_hash,
            "checkpoint_triplet": {
                "ree": state.pair["ree"], "q": state.pair["q"],
                "post": state.post_row,
            },
        }
        summary["base_trace_sha256"] = sha256_file(base_trace)
        summary["controller_trace_sha256"] = (
            sha256_file(controller_trace) if controller_trace.is_file() else None
        )
        summary["metrics"] = None
        if args.mode != "shadow":
            stats = list(output.rglob(
                f"stats_ep_ckpt_270_{args.split}_r0_w1.json"
            ))
            if len(stats) == 1:
                payload = json.loads(stats[0].read_text())
                summary["metrics"] = payload.get(str(args.episode_id))
                summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
