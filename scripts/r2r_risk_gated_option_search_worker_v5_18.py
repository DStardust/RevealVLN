#!/usr/bin/env python3
"""Risk-gated alternative-first search with V5.17 option elimination."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_remaining_set_rerank_worker_v5_17 as v517  # noqa: E402
from revealnav_scalar_failure_risk import (  # noqa: E402
    FEATURE_NAMES, ScalarETPFailureRiskHead,
)
from rxr_unseen_controller_worker import sha256_file  # noqa: E402


RISK_RESULT = ROOT / (
    "artifacts/phase1/r2r_scalar_failure_risk_v5_18_1/"
    "R2R_SCALAR_ETP_FAILURE_RISK_TRAINING.json"
)


class RiskGatedOptionSearchController(v517.RemainingSetRerankController):
    """Try a learned alternative first only when frozen ETP is high risk."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.risk_models, self.risk_threshold = self._load_risk_ensemble()
        self.risk_gate_evaluations = 0
        self.risk_gate_activations = 0
        self.risk_alternative_trials = 0
        self.risk_alternative_accepts = 0
        self.current_trial_is_risk_alternative = False
        self.current_trial_navigation_step: int | None = None

    def _load_risk_ensemble(self):
        result = json.loads(RISK_RESULT.read_text())
        evidence = result["checkpoint"]
        path = ROOT / evidence["path"]
        if not (
            result.get("status") == "R2R_SCALAR_FAILURE_RISK_PASS"
            and all(result.get("gates", {}).values())
            and result.get("dev_used_for_threshold_or_training") is False
            and result.get("unseen_or_test_read") is False
            and not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == evidence["bytes"]
            and sha256_file(path) == evidence["sha256"]
        ):
            raise RuntimeError("V5.18 scalar failure-risk closure drift")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        states = payload.get("model_state_dicts", ())
        if not (
            payload.get("schema_version")
            == "revealnav-scalar-etp-failure-risk-ensemble/1"
            and tuple(payload.get("member_seeds", ()))
            == (20260826, 20260827, 20260828)
            and payload.get("aggregation") == "mean_failure_probability"
            and tuple(payload.get("feature_names", ())) == FEATURE_NAMES
            and len(states) == 3
            and float(payload["threshold"]) == float(result["threshold"])
        ):
            raise RuntimeError("V5.18 scalar failure-risk payload drift")
        models = []
        for state in states:
            model = ScalarETPFailureRiskHead(
                state["mean"], state["scale"]
            ).to(self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            models.append(model)
        return tuple(models), float(payload["threshold"])

    def _failure_risk(
        self, native_branch: str, alternative: str,
    ) -> tuple[float, list[float], list[float]]:
        histories = [row[0].detach() for row in self.rows]
        if not histories:
            raise RuntimeError("risk gate lacks causal temporal history")
        graph = pilot._TRAINER.gmaps[0]
        checkpoint_id = pilot._CURRENT_IDS[0]
        if (
            checkpoint_id not in graph.node_pos
            or any(
                branch not in graph.ghost_aug_pos
                for branch in (native_branch, alternative)
            )
        ):
            raise RuntimeError("risk gate lacks online branch geometry")
        checkpoint = np.asarray(graph.node_pos[checkpoint_id], dtype=np.float32)
        distances = [
            float(np.linalg.norm(
                np.asarray(graph.ghost_aug_pos[branch], dtype=np.float32)
                - checkpoint
            ))
            for branch in (native_branch, alternative)
        ]
        inputs = (
            self.instruction.unsqueeze(0),
            self.latest_history.unsqueeze(0),
            torch.stack(histories).mean(0).unsqueeze(0),
            self.global_current[native_branch].unsqueeze(0),
            self.global_current[alternative].unsqueeze(0),
            torch.tensor(
                [[distances[0] / 10.0, distances[1] / 10.0]],
                dtype=torch.float32, device=self.device,
            ),
        )
        with torch.no_grad():
            probabilities = [
                float(torch.sigmoid(model(*inputs))[0])
                for model in self.risk_models
            ]
        return sum(probabilities) / len(probabilities), probabilities, distances

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        if not self._has_switch_budget(self.step, int(pilot._TRAINER.max_len)):
            return super()._start_native_trial(
                controls, native_branch, alternative, alternative_source, value
            )
        risk, members, distances = self._failure_risk(native_branch, alternative)
        self.risk_gate_evaluations += 1
        high_risk = risk > self.risk_threshold
        self.record(
            "pre_action_failure_risk",
            checkpoint_id=pilot._CURRENT_IDS[0],
            native_branch=native_branch,
            alternative_branch=alternative,
            alternative_source=alternative_source,
            ensemble_failure_probability=round(risk, 8),
            member_failure_probabilities=[round(value, 8) for value in members],
            frozen_threshold=round(self.risk_threshold, 8),
            high_risk=high_risk,
            native_distance_m=round(distances[0], 6),
            alternative_distance_m=round(distances[1], 6),
            causal_inputs_only=True,
        )
        if not high_risk:
            return super()._start_native_trial(
                controls, native_branch, alternative, alternative_source, value
            )
        self.risk_gate_activations += 1
        return self._start_risk_alternative_trial(
            controls, native_branch, alternative, alternative_source, value,
            risk,
        )

    def _start_risk_alternative_trial(
        self, controls, native_branch, alternative, alternative_source, value,
        risk: float,
    ):
        preservation_gain = float(value["macro"].preservation_gain)
        checkpoint_id = pilot._CURRENT_IDS[0]
        self.ledger.register(checkpoint_id, controls)
        if not self.ledger.authorize_branch(
            checkpoint_id, alternative, preservation_gain
        ):
            raise RuntimeError("risk-selected alternative is not ledger-authorized")
        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = self.global_current[alternative].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = alternative
        self.checkpoint_id = checkpoint_id
        graph = pilot._TRAINER.gmaps[0]
        self.checkpoint_graph_snapshot = copy.deepcopy(graph)
        self.checkpoint_graph_signature = self._graph_signature(graph)
        self.topology_snapshots += 1
        self.checkpoint_position = np.asarray(
            graph.node_pos[checkpoint_id], dtype=float
        ).copy()
        self.executor = v517.v516.StateConditionedReturnExecutor(
            checkpoint_id, "ETP-R1:frozen-control", tuple(controls)
        )
        self.executor.start_excursion(alternative)
        self.checkpoint_candidates = {
            branch: self.global_current[branch].detach() for branch in controls
        }
        self.trial_preservation_gain = preservation_gain
        self.checkpoint_control_ids = tuple(controls)
        self.exhausted_option_ids = set()
        self.search_histories = [value.detach() for value in self.pre_histories]
        self.search_rows = [
            (
                history.detach(),
                {key: embedding.detach() for key, embedding in candidates.items()},
            )
            for history, candidates in self.rows
        ]
        if len(self.search_rows) != len(self.search_histories):
            raise RuntimeError("risk trial temporal candidate rows are misaligned")
        self.latest_probe_row = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step = None
        self.current_trial_is_risk_alternative = True
        self.current_trial_navigation_step = self.step
        self.risk_alternative_trials += 1
        self.checkpointed_excursions += 1
        self.record(
            "risk_alternative_trial_created",
            checkpoint_id=checkpoint_id,
            trial_branch=alternative,
            retained_native_branch=native_branch,
            candidate_ids=list(controls),
            alternative_source=alternative_source,
            ensemble_failure_probability=round(risk, 8),
            preservation_gain=round(preservation_gain, 8),
            reversible=True,
            direct_irreversible_commit=False,
        )
        self.global_rows.clear()
        return alternative

    def _post_decision(self, current) -> None:
        risk_trial = self.current_trial_is_risk_alternative
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        navigation_step = self.current_trial_navigation_step
        before = len(self.events)
        super()._post_decision(current)
        if not risk_trial:
            return
        post = [
            event for event in self.events[before:]
            if event.get("event") == "post_decision"
        ]
        if len(post) != 1:
            raise RuntimeError("risk alternative lacks one post decision")
        if post[0].get("executed_return") is False:
            self.risk_alternative_accepts += 1
            self.alternative_commits += 1
            self.record(
                "risk_alternative_probe_accepted",
                checkpoint_id=checkpoint_id,
                branch_id=branch_id,
                navigation_step=navigation_step,
                acceptance="robust_post_Q_continue_or_REE_closed",
            )
            self.current_trial_is_risk_alternative = False
            self.current_trial_navigation_step = None
        elif self.pending_return_action is None:
            self.current_trial_is_risk_alternative = False
            self.current_trial_navigation_step = None

    def complete_pending_return(self) -> None:
        super().complete_pending_return()
        self.current_trial_is_risk_alternative = False
        self.current_trial_navigation_step = None

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "risk_gate_evaluations": self.risk_gate_evaluations,
            "risk_gate_activations": self.risk_gate_activations,
            "risk_alternative_trials": self.risk_alternative_trials,
            "risk_alternative_accepts": self.risk_alternative_accepts,
            "risk_gate_threshold": self.risk_threshold,
            "risk_gate_model": "16-scalar 17-parameter logistic ensemble",
            "intervention_contract": (
                "at a valid checkpoint, use a train-only causal scalar risk "
                "ensemble to choose alternative-first when frozen ETP failure "
                "risk exceeds its calibration threshold and native-first "
                "otherwise; all probes remain governed by robust post-Q, "
                "verified physical return, exact topology restoration, and "
                "frozen-ETP remaining-set reranking"
            ),
        })
        return value


def _validate_v5_18_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    checks = []
    for event in state.events:
        kind = event.get("event")
        if kind == "native_first_trial_created":
            expected, step = event["native_branch"], event["step"]
        elif kind == "risk_alternative_trial_created":
            expected, step = event["trial_branch"], event["step"]
        elif kind in (
            "remaining_set_rerank_committed", "remaining_set_probe_created",
        ):
            expected, step = event["branch_id"], event["navigation_step"]
        else:
            continue
        in_range = 0 <= step < len(actions)
        action = actions[step] if in_range else {}
        observed = action.get("ghost_vp")
        act = action.get("act")
        equal = in_range and (
            (expected is None and act == 0)
            or (expected is not None and act == 4 and observed == expected)
        )
        checks.append({
            "event": kind, "step": step, "expected_branch": expected,
            "executed_act": act, "executed_ghost_vp": observed,
            "in_range": in_range, "equal": equal,
        })
    if not all(row["equal"] for row in checks):
        raise RuntimeError("V5.18 declared/executed action identity mismatch")
    return {"checks": len(checks), "all_equal": True, "rows": checks}


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v517.RemainingSetRerankController = RiskGatedOptionSearchController
    v517.main()
    state = v517.V55._CONTROLLER
    if not isinstance(state, RiskGatedOptionSearchController):
        raise RuntimeError("V5.18 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.18"
    summary["method_revision"] = (
        "train-only scalar frozen-ETP failure risk chooses reversible "
        "alternative-first versus native-first option search"
    )
    summary["safety_funnel"] = state.safety_funnel()
    if summary.get("mode") == "revealnav":
        summary["executed_action_validation"] = _validate_v5_18_actions(
            state, run_dir / "base_trace.jsonl"
        )
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
