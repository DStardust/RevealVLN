#!/usr/bin/env python3
"""Collect one RxR-train V6.5 all-live decision-group counterfactual."""

from __future__ import annotations

import hashlib
import json
import math
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
import rxr_v6_2_counterfactual_worker as v62  # noqa: E402
import rxr_v6_counterfactual_worker as base  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402


SCOPE = {
    "auxiliary_mechanism_diagnostic_only": True,
    "not_vln_mainline": True,
    "cannot_gate_uad_mainline": True,
    "intended_downstream_task": "open_vocabulary_object_search",
    "public_rxr_r2r_unseen_authorized": False,
    "uad_mainline_training_authorized": False,
}


def update_array_hash(digest, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())


def candidate_set_sha256(candidates: dict[str, torch.Tensor]) -> str:
    """Hash canonical candidate identities and embedding content."""
    if len(candidates) < 3 or len(candidates) > 4:
        raise ValueError("V6.5 candidate width must be three or four")
    digest = hashlib.sha256()
    for branch_id in sorted(candidates):
        value = candidates[branch_id].detach().cpu().half().numpy()
        digest.update(json.dumps(str(branch_id)).encode())
        update_array_hash(digest, "embedding", value)
    return digest.hexdigest()


def shared_state_sha256(
    arrays: dict[str, np.ndarray], candidate_hash: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(candidate_hash.encode())
    for key in sorted(arrays):
        update_array_hash(digest, key, arrays[key])
    return digest.hexdigest()


def option_causal_sha256(
    shared_hash: str, branch_id: str, embedding: np.ndarray,
    scalars: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(shared_hash.encode())
    digest.update(json.dumps(str(branch_id)).encode())
    update_array_hash(digest, "option_embedding", embedding)
    update_array_hash(digest, "option_scalars", scalars)
    return digest.hexdigest()


class AllLiveAlternativeCounterfactualController(
    v62.LocalTopologyCandidateController
):
    """Freeze one native decision group and enumerate every live alternative."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v65_initial_evidence: dict | None = None

    @staticmethod
    def live_controls(
        current: dict, native: str | None, frontier_age: dict[str, int],
        status,
    ) -> tuple[str, ...]:
        """Return native plus every currently scoreable UNTRIED alternative."""
        if (
            native is None
            or native not in current
            or status(native) is not OptionStatus.UNTRIED
        ):
            return ()
        alternatives = sorted(
            (
                branch for branch in current
                if branch != native and status(branch) is OptionStatus.UNTRIED
            ),
            key=lambda branch: (-int(frontier_age[branch]), branch),
        )[:3]
        return (native, *alternatives) if len(alternatives) >= 2 else ()

    def proposal_controls(
        self, current: dict, native: str | None, frontier_age: dict[str, int],
    ) -> tuple[str, ...]:
        checkpoint_id = str(pilot._CURRENT_IDS[0])
        return self.live_controls(
            current, native, frontier_age,
            lambda branch: self.ledger.status(checkpoint_id, branch),
        )

    def ranked_alternative(self, value: dict, controls, native: str):
        candidate_ids = tuple(sorted(str(branch) for branch in controls))
        candidates = {
            branch: self.global_current[branch].detach()
            for branch in candidate_ids
        }
        probabilities = {
            str(branch): float(value["probabilities"][index])
            for index, branch in enumerate(controls)
        }
        alternatives = tuple(
            branch for branch in candidate_ids if branch != native
        )
        if len(alternatives) not in (2, 3):
            raise RuntimeError("V6.5 group lacks two or three alternatives")
        default = min(
            alternatives, key=lambda branch: (-probabilities[branch], branch)
        )
        decision_index = len(self.candidate_events)
        selected = default
        if (
            self.v6_mode == "macro" and self.target is not None
            and int(self.target["decision_index"]) == decision_index
        ):
            selected = str(self.target["alternative_branch_id"])
            if selected not in alternatives:
                raise RuntimeError("V6.5 target alternative left candidate set")
            if not (
                str(self.target["checkpoint_id"]) == str(pilot._CURRENT_IDS[0])
                and str(self.target["native_branch_id"]) == str(native)
                and self.target["candidate_set_sha256"]
                == candidate_set_sha256(candidates)
            ):
                raise RuntimeError("V6.5 target candidate set drifted before outbound")
        self.v65_initial_evidence = {
            "decision_index": decision_index,
            "candidate_branch_ids": list(candidate_ids),
            "alternative_branch_ids": list(alternatives),
            "candidate_set_sha256": candidate_set_sha256(candidates),
            "probabilities": probabilities,
            "belief": {key: float(number) for key, number in value["belief"].items()},
            "default_alternative_branch_id": default,
        }
        return selected

    def _group_arrays(
        self, current: dict[str, torch.Tensor], group_id: str,
    ) -> tuple[dict[str, np.ndarray], list[dict], str, str]:
        initial = self.v65_initial_evidence
        if initial is None:
            raise RuntimeError("V6.5 group lacks frozen initial evidence")
        candidate_ids = initial["candidate_branch_ids"]
        candidates = {
            branch: self.checkpoint_candidates[branch]
            for branch in candidate_ids
        }
        candidate_hash = candidate_set_sha256(candidates)
        if candidate_hash != initial["candidate_set_sha256"]:
            raise RuntimeError("V6.5 candidate embeddings drifted during outbound")
        alternative_ids = initial["alternative_branch_ids"]
        history = torch.stack([
            *self.pre_histories, self.latest_history.detach()
        ])
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        shared = {
            "instruction": self.instruction.detach().cpu().half().numpy(),
            "post_observation": local.detach().cpu().half().numpy(),
            "temporal_history": history.mean(0).detach().cpu().half().numpy(),
            "checkpoint": self.checkpoint_embedding.detach().cpu().half().numpy(),
            "native": self.selected_embedding.detach().cpu().half().numpy(),
        }
        shared_hash = shared_state_sha256(shared, candidate_hash)
        return_distance = self._return_path_length()
        if return_distance is None:
            raise RuntimeError("V6.5 group has no executable online return path")
        probabilities = initial["probabilities"]
        values = np.asarray(list(probabilities.values()), dtype=np.float64)
        entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-12))))
        entropy /= math.log(len(values))
        before = initial["belief"]
        after = self._post_ree_belief(current)
        common = (
            self.step / float(pilot._TRAINER.max_len),
            max(0.0, float(pilot._TRAINER.max_len) - self.step)
            / float(pilot._TRAINER.max_len),
            return_distance / 10.0,
            probabilities[self.selected_branch],
        )
        option_embeddings = []
        option_scalars = []
        options = []
        for index, branch in enumerate(alternative_ids):
            embedding = candidates[branch].detach().cpu().half().numpy()
            scalars = np.asarray([
                *common,
                probabilities[branch],
                probabilities[branch] - probabilities[self.selected_branch],
                entropy,
                len(candidate_ids) / 4.0,
                before["p_discriminable"],
                before["evidence"],
                before["maximum_target_probability"],
                before["reveal_hazard"],
                before["expiry_hazard"],
                after["p_discriminable"],
                after["evidence"],
                after["selected_target_probability"],
            ], dtype=np.float32)
            if scalars.shape != (16,) or not np.isfinite(scalars).all():
                raise RuntimeError("V6.5 option scalar schema drift")
            option_embeddings.append(embedding)
            option_scalars.append(scalars)
            options.append({
                "option_index": index,
                "option_id": (
                    f"{group_id}_o{index:02d}_{branch}"
                ),
                "alternative_branch_id": branch,
                "option_causal_sha256": option_causal_sha256(
                    shared_hash, branch, embedding, scalars
                ),
            })
        arrays = {
            **shared,
            "option_embeddings": np.stack(option_embeddings),
            "option_scalars": np.stack(option_scalars),
            "option_ids": np.asarray([
                option["option_id"] for option in options
            ], dtype=str),
        }
        return arrays, options, candidate_hash, shared_hash

    def _event(self, current: dict[str, torch.Tensor]) -> dict:
        index = len(self.candidate_events)
        group_id = (
            f"rxr_ep{self.metadata['episode_id']}_seed{self.seed}_"
            f"post{self.step:03d}_g{index:02d}"
        )
        arrays, options, candidate_hash, shared_hash = self._group_arrays(
            current, group_id
        )
        feature_path = self.event_dir / f"{group_id}.npz"
        if self.v6_mode == "shadow":
            base.atomic_npz(feature_path, arrays)
        value = {
            "group_id": group_id,
            "event_id": group_id,
            "decision_index": index,
            "event_index": index,
            **self.metadata,
            "controller_seed": self.seed,
            "post_navigation_step": self.step,
            "prefix_action_count": self.step,
            "checkpoint_id": self.checkpoint_id,
            "native_branch_id": self.selected_branch,
            "alternative_branch_id": self.retained_alternative,
            "default_alternative_branch_id": self.v65_initial_evidence[
                "default_alternative_branch_id"
            ],
            "candidate_branch_ids": self.v65_initial_evidence[
                "candidate_branch_ids"
            ],
            "live_alternative_branch_ids": self.v65_initial_evidence[
                "alternative_branch_ids"
            ],
            "candidate_set_sha256": candidate_hash,
            "shared_state_sha256": shared_hash,
            "causal_state_sha256": shared_hash,
            "causal_prefix_only": True,
            "online_return_path_length_m": round(
                float(arrays["option_scalars"][0, 2]) * 10.0, 6
            ),
            "options": options,
            "feature_path": (
                str(feature_path.relative_to(ROOT))
                if self.v6_mode == "shadow" else None
            ),
            "feature_bytes": (
                feature_path.stat().st_size
                if self.v6_mode == "shadow" else None
            ),
            "feature_sha256": (
                base.sha256_file(feature_path)
                if self.v6_mode == "shadow" else None
            ),
        }
        self.candidate_events.append(value)
        return value

    def _matches_target(self, group: dict) -> bool:
        if self.target is None:
            return False
        keys = (
            "decision_index", "group_id", "checkpoint_id",
            "native_branch_id", "alternative_branch_id",
            "candidate_set_sha256", "shared_state_sha256",
        )
        if not all(group.get(key) == self.target.get(key) for key in keys):
            return False
        option = next((
            row for row in group["options"]
            if row["alternative_branch_id"] == group["alternative_branch_id"]
        ), None)
        return option is not None and option["option_causal_sha256"] == self.target.get(
            "option_causal_sha256"
        )

    def _post_decision(self, current) -> None:
        group = self._event(current)
        target_index = (
            None if self.target is None else int(self.target["decision_index"])
        )
        if self.v6_mode == "macro" and group["decision_index"] == target_index:
            if not self._matches_target(group):
                raise RuntimeError("V6.5 replay target group or option drift")
            self.target_reached = True
            self.post_policy_action = "backtrack"
            self.record(
                "v6_5_multi_option_intervention",
                decision_index=group["decision_index"],
                group_id=group["group_id"],
                shared_state_sha256=group["shared_state_sha256"],
                candidate_set_sha256=group["candidate_set_sha256"],
                native_branch=group["native_branch_id"],
                alternative_branch=group["alternative_branch_id"],
                executed_return=True,
                gate_mode="forced_train_counterfactual_only",
            )
            self.backtrack_decisions += 1
            self.target_return_scheduled = self._schedule_return()
            if not self.target_return_scheduled:
                raise RuntimeError("V6.5 target return could not be scheduled")
            return
        if (
            self.v6_mode == "macro" and target_index is not None
            and group["decision_index"] > target_index
            and not self.target_reached
        ):
            raise RuntimeError("V6.5 replay passed target without matching")
        self._continue_native(group)


def role_and_run_dir(argv: list[str]) -> tuple[str, Path, list[str]]:
    values = list(argv)
    if "--role" not in values:
        raise SystemExit("V6.5 worker requires --role")
    index = values.index("--role")
    if index + 1 >= len(values) or values[index + 1] not in (
        "development", "holdout"
    ):
        raise SystemExit("invalid V6.5 role")
    role = values[index + 1]
    del values[index:index + 2]
    if "--run-dir" not in values:
        raise SystemExit("V6.5 worker requires --run-dir")
    run_dir = Path(values[values.index("--run-dir") + 1]).resolve()
    return role, run_dir, values


def trace_macro_observation(value: dict, run_dir: Path) -> dict:
    """Derive target identity and return lifecycle from the actual trace."""
    target = value.get("target")
    if value.get("mode") != "macro" or not isinstance(target, dict):
        return {
            "observed_group_id": None,
            "observed_checkpoint_id": None,
            "observed_native_branch_id": None,
            "observed_candidate_branch_ids": None,
            "observed_candidate_set_sha256": None,
            "observed_shared_state_sha256": None,
            "observed_option_causal_sha256": None,
            "target_physical_return_verified": False,
            "target_topology_restored": False,
            "target_alternative_committed": False,
            "committed_alternative_branch_id": None,
        }
    group = next((
        row for row in value.get("candidate_events", [])
        if row.get("decision_index") == target.get("decision_index")
    ), None)
    option = None if group is None else next((
        row for row in group.get("options", [])
        if row.get("option_id") == target.get("option_id")
    ), None)
    observed = {
        "observed_group_id": None if group is None else group.get("group_id"),
        "observed_checkpoint_id": (
            None if group is None else group.get("checkpoint_id")
        ),
        "observed_native_branch_id": (
            None if group is None else group.get("native_branch_id")
        ),
        "observed_candidate_branch_ids": (
            None if group is None else group.get("candidate_branch_ids")
        ),
        "observed_candidate_set_sha256": (
            None if group is None else group.get("candidate_set_sha256")
        ),
        "observed_shared_state_sha256": (
            None if group is None else group.get("shared_state_sha256")
        ),
        "observed_option_causal_sha256": (
            None if option is None else option.get("option_causal_sha256")
        ),
        "target_physical_return_verified": False,
        "target_topology_restored": False,
        "target_alternative_committed": False,
        "committed_alternative_branch_id": None,
    }
    trace_path = run_dir / "controller_trace.jsonl"
    if not trace_path.is_file() or trace_path.is_symlink():
        return observed
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    phase = 0
    for row in rows:
        event = row.get("event")
        if phase == 0 and (
            event == "v6_5_multi_option_intervention"
            and row.get("group_id") == target.get("group_id")
            and row.get("native_branch") == target.get("native_branch_id")
            and row.get("alternative_branch")
            == target.get("alternative_branch_id")
            and row.get("candidate_set_sha256")
            == target.get("candidate_set_sha256")
            and row.get("shared_state_sha256")
            == target.get("shared_state_sha256")
        ):
            phase = 1
        elif phase == 1 and event == "return_complete" and row.get("success") is True:
            phase = 2
        elif phase == 2 and (
            event == "retained_alternative_armed"
            and row.get("checkpoint_id") == target.get("checkpoint_id")
            and row.get("rejected_native_branch") == target.get("native_branch_id")
            and row.get("branch_id") == target.get("alternative_branch_id")
            and row.get("return_verified") is True
        ):
            observed["target_physical_return_verified"] = True
            phase = 3
        elif phase == 3 and (
            event == "checkpoint_topology_restored"
            and row.get("checkpoint_id") == target.get("checkpoint_id")
        ):
            observed["target_topology_restored"] = True
            phase = 4
        elif phase == 4 and (
            event == "retained_alternative_committed"
            and row.get("checkpoint_id") == target.get("checkpoint_id")
            and row.get("rejected_native_branch") == target.get("native_branch_id")
            and row.get("branch_id") == target.get("alternative_branch_id")
        ):
            observed["target_alternative_committed"] = True
            observed["committed_alternative_branch_id"] = row["branch_id"]
            phase = 5
    return observed


def rewrite_summary(run_dir: Path, role: str) -> None:
    path = run_dir / "RUN_SUMMARY.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text())
    groups = value.pop("candidate_events", [])
    value.pop("candidate_event_count", None)
    value["candidate_events"] = groups
    observation = trace_macro_observation(value, run_dir)
    value.pop("candidate_events")
    value.update({
        "schema_version": "revealnav-rxr-v6.5-all-live-worker/1",
        "role": role,
        "decision_group_count": len(groups),
        "decision_groups": groups,
        "all_alternatives_enumerated_before_outcome": True,
        "branch_id_used_as_model_input": False,
        **observation,
        **SCOPE,
    })
    if value.get("mode") == "macro":
        complete = all((
            observation["target_physical_return_verified"],
            observation["target_topology_restored"],
            observation["target_alternative_committed"],
        ))
        if value.get("status") == "REJECTED_UNEXECUTABLE_MACRO":
            value["status"] = "REJECTED_UNEXECUTABLE_OPTION"
        elif value.get("status") == "PASS" and not complete:
            value["status"] = "FAIL"
            value["error"] = "trace does not prove the complete target lifecycle"
    base.atomic_json(path, value)


def main() -> int:
    role, run_dir, forwarded = role_and_run_dir(sys.argv)
    sys.argv = forwarded
    base.V6CounterfactualController = AllLiveAlternativeCounterfactualController
    try:
        result = base.run()
    except BaseException:
        rewrite_summary(run_dir, role)
        raise
    rewrite_summary(run_dir, role)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
