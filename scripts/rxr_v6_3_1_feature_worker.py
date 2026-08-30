#!/usr/bin/env python3
"""Replay V6.3.1 features with outer-fold-clean post-Q evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rxr_v6_3_feature_worker as v63  # noqa: E402
import rxr_v6_counterfactual_worker as base  # noqa: E402
from revealnav_mf2r4 import PostExcursionQHead  # noqa: E402
from revealnav_mf2r6.protocol import (  # noqa: E402
    outer_scene_partition, scene_fold, stable_hash,
)


POST_Q_ROOT = ROOT / "artifacts/phase1/rxr_v6/v6_3_1/post_q_outer"
V6_SELECTION = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_2/RXR_V6_EPISODE_SELECTION.json"
)


class CrossFitCausalEvidenceController(v63.CausalEvidenceController):
    def __init__(self, seed, trace_path, event_dir, metadata, mode, target):
        self.v631_scene_fold = scene_fold(str(metadata["scene_id"]))
        super().__init__(seed, trace_path, event_dir, metadata, mode, target)

    def _load_post_ensemble(self):
        selection = json.loads(V6_SELECTION.read_text())
        selection_sha256 = base.sha256_file(V6_SELECTION)
        scenes = {str(row["scene_id"]) for row in selection["episodes"]}
        models_by_fold = {}
        evidence_by_fold = {}
        for fold in range(5):
            roles = outer_scene_partition(scenes, fold)
            role_scenes = {
                role: sorted(scene for scene, value in roles.items()
                             if value == role)
                for role in ("fit", "calibration", "evaluation")
            }
            result_path = POST_Q_ROOT / f"fold_{fold}/RESULT.json"
            if result_path.is_symlink() or not result_path.is_file():
                raise RuntimeError("missing V6.3.1 post-Q outer result")
            result = json.loads(result_path.read_text())
            checkpoint = (ROOT / result["checkpoint"]["path"]).resolve()
            sources_valid = True
            for relative, evidence in result.get("sources", {}).items():
                source = (ROOT / relative).resolve()
                sources_valid = sources_valid and (
                    ROOT in source.parents
                    and not source.is_symlink()
                    and source.is_file()
                    and source.stat().st_size == evidence.get("bytes")
                    and base.sha256_file(source) == evidence.get("sha256")
                )
            role_hashes_match = all(
                result.get(f"{role}_v6_scene_ids_sha256")
                == stable_hash(role_scenes[role])
                and result.get(f"{role}_v6_scene_count")
                == len(role_scenes[role])
                for role in role_scenes
            )
            if not (
                result.get("status") == "V6_3_1_POST_Q_OUTER_READY"
                and result.get("outer_fold") == fold
                and result.get(
                    "calibration_or_evaluation_scene_leakage"
                ) is False
                and role_hashes_match
                and result.get("sources", {}).get(
                    str(V6_SELECTION.relative_to(ROOT)), {}
                ).get("sha256") == selection_sha256
                and sources_valid
                and ROOT in checkpoint.parents
                and not checkpoint.is_symlink() and checkpoint.is_file()
                and checkpoint.stat().st_size == result["checkpoint"]["bytes"]
                and base.sha256_file(checkpoint)
                == result["checkpoint"]["sha256"]
            ):
                raise RuntimeError("V6.3.1 post-Q outer evidence drift")
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            states = payload.get("model_state_dicts", ())
            if not (
                payload.get("schema_version")
                == "revealnav-v6.3.1-post-q-outer-ensemble/1"
                and payload.get("outer_fold") == fold
                and payload.get("v6_selection_sha256") == selection_sha256
                and all(
                    payload.get(f"{role}_v6_scene_ids_sha256")
                    == stable_hash(role_scenes[role])
                    for role in role_scenes
                )
                and len(states) == 3
            ):
                raise RuntimeError("V6.3.1 post-Q outer checkpoint drift")
            models = []
            for state in states:
                model = PostExcursionQHead(768, 96, 5.0)
                model.load_state_dict(state, strict=True)
                models.append(model.to(self.device).eval())
            models_by_fold[fold] = tuple(models)
            evidence_by_fold[str(fold)] = {
                "outer_fold": fold,
                "result_path": str(result_path.relative_to(ROOT)),
                "result_bytes": result_path.stat().st_size,
                "result_sha256": base.sha256_file(result_path),
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": base.sha256_file(checkpoint),
                **{
                    f"{role}_v6_scene_ids_sha256": stable_hash(
                        role_scenes[role]
                    )
                    for role in role_scenes
                },
            }
        self.v631_post_models_by_fold = models_by_fold
        self.v631_post_q_evidence = evidence_by_fold
        return models_by_fold[self.v631_scene_fold]

    def _causal_arrays(self, current):
        arrays = v63.v62.LocalTopologyCandidateController._causal_arrays(
            self, current
        )
        initial = self.v63_initial_evidence
        if initial is None:
            raise RuntimeError("V6.3.1 event lacks initial causal evidence")
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        ).unsqueeze(0)
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        inputs = (
            history,
            torch.tensor([history.shape[1]], device=self.device),
            self.instruction.unsqueeze(0),
            self.selected_embedding.unsqueeze(0),
            self.checkpoint_embedding.unsqueeze(0),
            local.unsqueeze(0),
            torch.tensor([1.0], device=self.device),
        )
        fold_scalars = []
        post_belief = self._post_ree_belief(current)
        with torch.no_grad():
            for fold in range(5):
                margins = []
                for model in self.v631_post_models_by_fold[fold]:
                    output = model(*inputs)
                    margins.append(float(
                        output.continue_cost[0] - output.backtrack_cost[0]
                    ))
                extra = v63.evidence_scalars(
                    initial["probabilities"], initial["native"],
                    initial["alternative"], initial["belief"],
                    initial["preservation_gain"], margins,
                    post_belief,
                )
                fold_scalars.append(np.concatenate((arrays["scalars"], extra)))
        arrays["outer_fold_scalars"] = np.stack(fold_scalars).astype(
            np.float32, copy=False
        )
        self.v631_last_arrays = arrays
        return arrays

    def _event(self, current):
        value = super()._event(current)
        arrays = self.v631_last_arrays
        base_arrays = {
            key: array for key, array in arrays.items()
            if key != "outer_fold_scalars"
        }
        value["base_causal_state_sha256"] = base.stable_array_hash(base_arrays)
        value["post_q_outer_evidence"] = dict(self.v631_post_q_evidence)
        value["runtime_scene_fold"] = self.v631_scene_fold
        return value


def main() -> int:
    base.V6CounterfactualController = CrossFitCausalEvidenceController
    return base.run()


if __name__ == "__main__":
    raise SystemExit(main())
