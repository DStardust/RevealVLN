#!/usr/bin/env python3
"""Replay V6.2 shadows with causal decision evidence attached."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rxr_v6_1_counterfactual_worker as v61  # noqa: E402
import rxr_v6_2_counterfactual_worker as v62  # noqa: E402
import rxr_v6_counterfactual_worker as base  # noqa: E402


def evidence_scalars(
    probabilities: dict[str, float], native: str, alternative: str,
    initial_belief: dict[str, float], preservation_gain: float,
    post_q_margins: list[float], post_belief: dict[str, float],
) -> np.ndarray:
    """Return the fixed V6.3 online-evidence vector."""
    if not post_q_margins:
        raise ValueError("V6.3 requires post-Q margins")
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    if native not in probabilities or alternative not in probabilities:
        raise ValueError("V6.3 probability map lacks a selected branch")
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-12))))
    entropy /= math.log(len(values)) if len(values) > 1 else 1.0
    margins = np.asarray(post_q_margins, dtype=np.float64)
    result = np.asarray([
        probabilities[native], probabilities[alternative],
        probabilities[alternative] - probabilities[native], entropy,
        initial_belief["p_discriminable"], initial_belief["evidence"],
        initial_belief["maximum_target_probability"],
        initial_belief["reveal_hazard"], initial_belief["expiry_hazard"],
        preservation_gain,
        float(margins.mean()), float(margins.min()), float(margins.std()),
        float(np.mean(margins > 0.0)),
        post_belief["p_discriminable"], post_belief["evidence"],
        post_belief["selected_target_probability"],
    ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite V6.3 causal evidence")
    return result


class CausalEvidenceController(v62.LocalTopologyCandidateController):
    """Attach already-computed causal policy evidence to each V6 event."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v63_initial_evidence: dict | None = None

    def ranked_alternative(self, value: dict, controls, native: str):
        alternative = v61.BroadPersistentCandidateController.ranked_alternative(
            value, controls, native
        )
        if alternative is None:
            self.v63_initial_evidence = None
            return None
        probabilities = {
            branch: float(value["probabilities"][index])
            for index, branch in enumerate(controls)
        }
        self.v63_initial_evidence = {
            "probabilities": probabilities,
            "native": native,
            "alternative": alternative,
            "belief": dict(value["belief"]),
            "preservation_gain": float(value["macro"].preservation_gain),
        }
        return alternative

    def _causal_arrays(self, current: dict[str, torch.Tensor]):
        arrays = super()._causal_arrays(current)
        initial = self.v63_initial_evidence
        if initial is None:
            raise RuntimeError("V6.3 event lacks initial causal evidence")
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
        margins = []
        with torch.no_grad():
            for model in self.post_models:
                output = model(*inputs)
                margins.append(float(
                    output.continue_cost[0] - output.backtrack_cost[0]
                ))
        extra = evidence_scalars(
            initial["probabilities"], initial["native"],
            initial["alternative"], initial["belief"],
            initial["preservation_gain"], margins,
            self._post_ree_belief(current),
        )
        arrays["scalars"] = np.concatenate((arrays["scalars"], extra))
        return arrays


def main() -> int:
    base.V6CounterfactualController = CausalEvidenceController
    return base.run()


if __name__ == "__main__":
    raise SystemExit(main())
