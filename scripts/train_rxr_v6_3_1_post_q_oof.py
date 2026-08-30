#!/usr/bin/env python3
"""Train outer-fold post-Q ensembles on V6.3.1 fit scenes only."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rxr_post_excursion_q_v4_8 as base  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    PostExcursionDataset, PostExcursionQHead, PostExcursionQLoss,
    collate_post_excursion_examples,
)
from revealnav_mf2r6.protocol import (  # noqa: E402
    outer_scene_partition, stable_hash,
)


MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/post_excursion_v4_7/"
    "RXR_POST_EXCURSION_FULL_MANIFEST_V4_7.json"
)
DATA_RESULT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/post_excursion_v4_7/"
    "RXR_POST_EXCURSION_FULL_RESULT_V4_7.json"
)
V6_SELECTION = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_2/RXR_V6_EPISODE_SELECTION.json"
)
V6_MANIFEST = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_2/"
    "RXR_V6_PAIRED_DATASET_MANIFEST.json"
)
DESIGN = ROOT / (
    "artifacts/design/MF2_POLICY_RELATIVE_REVERSIBLE_ADVANTAGE_V6_3_1.md"
)
POST_Q_MODEL = ROOT / "revealnav_mf2r4/post_excursion.py"
POST_Q_DATA = ROOT / "revealnav_mf2r4/post_excursion_data.py"
PARTITION_SOURCE = ROOT / "revealnav_mf2r6/protocol.py"
OUTPUT = ROOT / "artifacts/phase1/rxr_v6/v6_3_1/post_q_outer"
SEEDS = (20260826, 20260827, 20260828)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def selected(
    dataset: PostExcursionDataset, fit_scenes: set[str],
) -> Subset:
    indices = [
        index for index, example in enumerate(dataset.examples)
        if str(example[0]["scene_id"]) in fit_scenes
    ]
    if not indices:
        raise RuntimeError("empty V6.3.1 post-Q partition")
    return Subset(dataset, indices)


def subset_identity(dataset: PostExcursionDataset, subset: Subset) -> str:
    rows = [
        {
            "event_id": str(dataset.examples[index][0]["event_id"]),
            "scene_id": str(dataset.examples[index][0]["scene_id"]),
            "branch_index": int(dataset.examples[index][3]),
        }
        for index in subset.indices
    ]
    return stable_hash(rows)


def train_member(
    train: Subset, development: Subset, seed: int, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PostExcursionQHead(768, 96, 5.0).to(device)
    objective = PostExcursionQLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=32, shuffle=True, generator=generator,
        collate_fn=collate_post_excursion_examples,
    )
    development_loader = DataLoader(
        development, batch_size=32, shuffle=False,
        collate_fn=collate_post_excursion_examples,
    )
    best_loss = None
    best_state = None
    stale = 0
    epochs = 0
    for epoch in range(1, 31):
        epochs = epoch
        model.train()
        for cpu in train_loader:
            batch = base.move(cpu, device)
            loss = objective(base.forward(model, batch), batch)["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V6.3.1 post-Q loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = base.move(cpu, device)
                loss = objective(base.forward(model, batch), batch)["total"]
                size = int(batch["continue_cost"].shape[0])
                total += float(loss) * size
                count += size
        value = total / count
        if best_loss is None or value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is None:
        raise RuntimeError("V6.3.1 post-Q produced no checkpoint")
    return best_state, {
        "seed": seed, "epochs": epochs,
        "best_development_loss": float(best_loss),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    fold = args.fold
    output = OUTPUT / f"fold_{fold}"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    source_train = PostExcursionDataset(MANIFEST, "train")
    source_development = PostExcursionDataset(MANIFEST, "development")
    data_result = json.loads(DATA_RESULT.read_text())
    selection = json.loads(V6_SELECTION.read_text())
    if not (
        data_result.get("status") == "POST_EXCURSION_FULL_GATE_PASS"
        and data_result.get("training_authorized") is True
        and selection.get("split") == "train"
        and selection.get("episode_count") == len(selection.get("episodes", ()))
        and selection.get("unseen_or_test_read") is False
    ):
        raise RuntimeError("V6.3.1 upstream evidence is not authorized")
    v6_scenes = {str(row["scene_id"]) for row in selection["episodes"]}
    roles = outer_scene_partition(v6_scenes, fold)
    fit_scenes = {scene for scene, role in roles.items() if role == "fit"}
    calibration_scenes = {
        scene for scene, role in roles.items() if role == "calibration"
    }
    evaluation_scenes = {
        scene for scene, role in roles.items() if role == "evaluation"
    }
    train = selected(source_train, fit_scenes)
    development = selected(source_development, fit_scenes)
    train_scenes = {
        source_train.examples[index][0]["scene_id"] for index in train.indices
    }
    development_scenes = {
        source_development.examples[index][0]["scene_id"]
        for index in development.indices
    }
    if not (train_scenes | development_scenes) <= fit_scenes:
        raise RuntimeError("non-fit V6 scene entered post-Q training")
    if (train_scenes | development_scenes) & (
        calibration_scenes | evaluation_scenes
    ):
        raise RuntimeError("outer calibration/evaluation scene leakage")
    if train_scenes & development_scenes:
        raise RuntimeError("post-Q train/development scene leakage")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    states = []
    members = []
    for seed in SEEDS:
        state, evidence = train_member(
            train, development, seed + fold * 100, device
        )
        states.append(state)
        members.append(evidence)
    output.mkdir(parents=True)
    checkpoint = output / "post_q_outer_ensemble.pt"
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save({
        "schema_version": "revealnav-v6.3.1-post-q-outer-ensemble/1",
        "outer_fold": fold,
        "member_base_seeds": list(SEEDS),
        "member_effective_seeds": [seed + fold * 100 for seed in SEEDS],
        "model_state_dicts": states,
        "feature_dim": 768, "hidden_dim": 96,
        "elapsed_denominator": 5.0,
        "source_manifest_sha256": base.sha256_file(MANIFEST),
        "source_result_sha256": base.sha256_file(DATA_RESULT),
        "v6_selection_sha256": base.sha256_file(V6_SELECTION),
        "parent_v6_manifest_sha256": base.sha256_file(V6_MANIFEST),
        "fit_v6_scene_ids_sha256": stable_hash(sorted(fit_scenes)),
        "calibration_v6_scene_ids_sha256": stable_hash(
            sorted(calibration_scenes)
        ),
        "evaluation_v6_scene_ids_sha256": stable_hash(
            sorted(evaluation_scenes)
        ),
        "train_example_identity_sha256": subset_identity(source_train, train),
        "development_example_identity_sha256": subset_identity(
            source_development, development
        ),
    }, part)
    os.replace(part, checkpoint)
    value = {
        "schema_version": "revealnav-v6.3.1-post-q-outer-result/1",
        "status": "V6_3_1_POST_Q_OUTER_READY",
        "outer_fold": fold,
        "train_examples": len(train),
        "development_examples": len(development),
        "train_scenes": len(train_scenes),
        "development_scenes": len(development_scenes),
        "train_development_scene_overlap": 0,
        "calibration_or_evaluation_scene_leakage": False,
        "fit_v6_scene_count": len(fit_scenes),
        "fit_v6_scene_ids_sha256": stable_hash(sorted(fit_scenes)),
        "calibration_v6_scene_count": len(calibration_scenes),
        "calibration_v6_scene_ids_sha256": stable_hash(
            sorted(calibration_scenes)
        ),
        "evaluation_v6_scene_count": len(evaluation_scenes),
        "evaluation_v6_scene_ids_sha256": stable_hash(
            sorted(evaluation_scenes)
        ),
        "train_scene_ids_sha256": stable_hash(sorted(train_scenes)),
        "development_scene_ids_sha256": stable_hash(
            sorted(development_scenes)
        ),
        "train_example_identity_sha256": subset_identity(source_train, train),
        "development_example_identity_sha256": subset_identity(
            source_development, development
        ),
        "members": members,
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": base.sha256_file(checkpoint),
        },
        "sources": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            }
            for path in (
                MANIFEST, DATA_RESULT, V6_SELECTION, V6_MANIFEST, DESIGN,
                POST_Q_MODEL, POST_Q_DATA, PARTITION_SOURCE,
                Path(__file__).resolve(),
            )
        },
        "v6_advantage_labels_semantically_read": False,
        "parent_v6_manifest_hash_only": True,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    atomic_json(output / "RESULT.json", value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
