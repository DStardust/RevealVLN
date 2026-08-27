import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from revealnav_mf2 import (
    RevealFeatureDataset,
    RevealOptionHeads,
    RevealOptionLoss,
    RevealOptionLossConfig,
    collate_reveal_examples,
    select_topk_options,
)


class Top2TensorTest(unittest.TestCase):
    def test_exhausted_option_promotes_the_next_candidate(self) -> None:
        loss = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        mask = torch.ones_like(loss, dtype=torch.bool)
        first, valid = select_topk_options(loss, mask)
        self.assertEqual(first.tolist(), [[0, 1]])
        self.assertTrue(valid.all())
        exhausted = torch.tensor([[True, False, False, False]])
        promoted, valid = select_topk_options(loss, mask, exhausted)
        self.assertEqual(promoted.tolist(), [[1, 2]])
        self.assertTrue(valid.all())

    def test_masked_candidate_is_never_selected(self) -> None:
        loss = torch.tensor([[0.0, 0.2, 0.3]])
        mask = torch.tensor([[False, True, True]])
        selected, _ = select_topk_options(loss, mask)
        self.assertEqual(selected.tolist(), [[1, 2]])


class RevealOptionHeadsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = RevealOptionHeads(feature_dim=16, hidden_dim=12)
        self.model.eval()
        self.history = torch.randn(2, 5, 16)
        self.candidates = torch.randn(2, 5, 4, 16)
        self.mask = torch.tensor([
            [[1, 1, 1, 0]] * 5,
            [[1, 1, 1, 1]] * 5,
        ], dtype=torch.bool)
        self.budgets = torch.tensor([1.5, 2.0, 3.0, 4.0]).view(
            1, 1, 4
        ).expand(2, 5, 4)
        self.instruction = torch.randn(2, 16)

    def test_shapes_and_causal_future_invariance(self) -> None:
        first = self.model(
            self.history, self.candidates, self.mask, self.budgets,
            self.instruction,
        )
        changed_history = self.history.clone()
        changed_candidates = self.candidates.clone()
        changed_history[:, 3:] += 100.0
        changed_candidates[:, 3:] -= 100.0
        second = self.model(
            changed_history, changed_candidates, self.mask, self.budgets,
            self.instruction,
        )
        self.assertEqual(first.option_cost.shape, (2, 5, 4))
        self.assertEqual(first.current_feasibility_logits.shape, (2, 5, 4, 4))
        self.assertTrue(torch.equal(
            first.option_cost[:, :3], second.option_cost[:, :3]
        ))
        self.assertTrue(torch.isinf(first.option_cost[0, :, 3]).all())

    def test_loss_is_finite_and_backpropagates(self) -> None:
        self.model.train()
        output = self.model(
            self.history, self.candidates, self.mask, self.budgets,
            self.instruction,
        )
        costs = torch.rand(2, 5, 4) * 3.0
        costs[~self.mask] = torch.inf
        batch = {
            "candidate_mask": self.mask,
            "target_index": torch.zeros(2, 5, dtype=torch.long),
            "target_in_set": torch.ones(2, 5),
            "separation": torch.ones(2, 5),
            "evidence_complete": torch.ones(2, 5),
            "reveal_hazard": torch.zeros(2, 5),
            "option_cost": costs,
            "current_feasibility": torch.ones(2, 5, 4, 4),
            "checkpoint_value": torch.rand(2, 5),
        }
        losses = RevealOptionLoss()(output, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ))

    def test_balanced_state_bce_accepts_four_positive_weights(self) -> None:
        objective = RevealOptionLoss(RevealOptionLossConfig(
            state_pos_weights=(0.25, 0.5, 2.0, 4.0)
        ))
        self.assertEqual(
            objective.config.state_pos_weights, (0.25, 0.5, 2.0, 4.0)
        )
        with self.assertRaises(ValueError):
            RevealOptionLoss(RevealOptionLossConfig(
                state_pos_weights=(1.0, 1.0, 0.0, 1.0)
            ))


class RevealFeatureDatasetTest(unittest.TestCase):
    @staticmethod
    def arrays(steps: int, candidates: int) -> dict[str, np.ndarray]:
        mask = np.ones((steps, candidates), dtype=np.bool_)
        return {
            "instruction_embedding": np.zeros(16, dtype=np.float32),
            "history_embeddings": np.zeros((steps, 16), dtype=np.float32),
            "candidate_embeddings": np.zeros(
                (steps, candidates, 16), dtype=np.float32
            ),
            "candidate_mask": mask,
            "target_index": np.zeros(steps, dtype=np.int64),
            "target_in_set": np.ones(steps, dtype=np.float32),
            "separation": np.ones(steps, dtype=np.float32),
            "evidence_complete": np.ones(steps, dtype=np.float32),
            "reveal_hazard": np.zeros(steps, dtype=np.float32),
            "option_cost": np.ones((steps, candidates), dtype=np.float32),
            "current_feasibility": np.ones(
                (steps, candidates, 4), dtype=np.float32
            ),
            "checkpoint_value": np.zeros(steps, dtype=np.float32),
        }

    def test_safe_npz_loading_and_variable_candidate_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index, (steps, candidates) in enumerate(((3, 2), (5, 4))):
                path = root / f"sample_{index}.npz"
                np.savez(path, **self.arrays(steps, candidates))
                raw = path.read_bytes()
                records.append({
                    "event_id": f"e{index}",
                    "scene_id": f"s{index}",
                    "split": "train",
                    "path": path.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "revealnav-mf2-feature-manifest/1",
                "records": records,
                "metadata": {},
            }))
            dataset = RevealFeatureDataset(manifest, "train")
            batch = collate_reveal_examples([dataset[0], dataset[1]])
            self.assertEqual(batch["candidate_embeddings"].shape, (2, 5, 4, 16))
            self.assertEqual(batch["candidate_mask"].sum().item(), 26)
            self.assertTrue(torch.isinf(batch["option_cost"][0, 3:]).all())
            self.assertTrue((batch["target_index"][0, 3:] == -1).all())


if __name__ == "__main__":
    unittest.main()
