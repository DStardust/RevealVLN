#!/usr/bin/env python3
"""Train and gate the causal paired-Q adapter for learned OPP."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import (  # noqa: E402
    CausalPairedQAdapter,
    CausalPairedQAdapterLoss,
    RevealExpiryQFeatureDataset,
    collate_reveal_expiry_q_examples,
)
from run_rxr_representation_comparison_v2 import move_batch  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import (  # noqa: E402
    atomic_json, rank_auc, scalar_train_medians, sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair/"
    "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
)
R3_1 = ROOT / "artifacts/evaluation/mf2_expiry_r3_1"
R3_1_COMPARISON = R3_1 / "RXR_EXPIRY_R3_COMPARISON.json"
R3_2_COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_opp_q_r3_2/RXR_OPP_Q_R3_2_COMPARISON.json"
)
REVISION = ROOT / (
    "artifacts/design/MF2_CAUSAL_PAIRED_Q_ADAPTER_CORRECTION_R3_3.md"
)
OUT = ROOT / "artifacts/evaluation/mf2_causal_opp_q_r3_3"
PROTOCOL = OUT / "RXR_CAUSAL_OPP_Q_R3_3_PROTOCOL.json"
COMPARISON = OUT / "RXR_CAUSAL_OPP_Q_R3_3_COMPARISON.json"


def build_protocol() -> dict:
    r31 = json.loads(R3_1_COMPARISON.read_text())
    r32 = json.loads(R3_2_COMPARISON.read_text())
    if not (
        r31.get("status") == "EXPIRY_R3_1_GATE_PASS"
        and r32.get("status") == "OPP_Q_R3_2_GATE_FAIL"
        and r31.get("gold_payload_read") is False
        and r32.get("gold_payload_read") is False
    ):
        raise RuntimeError("R3.3 precondition failed")
    checkpoints = {}
    for seed in SEEDS:
        path = R3_1 / f"augmented_seed_{seed}/relational_expiry_ree.pt"
        checkpoints[str(path.relative_to(ROOT))] = sha256_file(path)
    return {
        "schema_version": "revealnav-mf2-causal-opp-q-protocol/3.3",
        "status": "SEALED_BEFORE_CAUSAL_PAIRED_Q_R3_3_TRAINING",
        "seeds": list(SEEDS), "condition": "augmented",
        "architecture": {
            "feature_dim": 768, "hidden_dim": 96,
            "age_denominator": 128.0,
            "delta_activation": "relu_exact_zero",
            "causal": True,
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "epoch_limit": 20, "early_stopping_patience": 4,
        "loss": {"q_with_huber": 1.0, "q_without_huber": 1.0,
                 "paired_delta_huber": 1.0,
                 "within_prefix_margin_ranking": 0.25, "margin": 0.1},
        "selection": "minimum development native paired-Q loss",
        "success_gates": {
            "mean_q_with_mae_beats_train_median": True,
            "mean_q_without_mae_beats_train_median": True,
            "best_option_accuracy_above_random_in_two_seeds": True,
            "opv_mae_beats_zero_in_two_seeds": True,
            "opv_auc_above_0_5_in_two_seeds": True,
            "q_order_invariant": True,
            "source_r3_1_checkpoints_unchanged": True,
        },
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(R3_1_COMPARISON.relative_to(ROOT)): sha256_file(R3_1_COMPARISON),
            str(R3_2_COMPARISON.relative_to(ROOT)): sha256_file(R3_2_COMPARISON),
            str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
        },
        "frozen_r3_1_checkpoints": checkpoints,
        "gold_access_allowed": False,
        "additional_hyperparameter_search_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R3.3 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "seeds": value["seeds"],
                      "protocol": str(PROTOCOL.relative_to(ROOT)),
                      "sha256": sha256_file(PROTOCOL)}, indent=2))
    return 0


def loaders(seed: int):
    train = RevealExpiryQFeatureDataset(MANIFEST, "train")
    development = RevealExpiryQFeatureDataset(MANIFEST, "development")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=8, shuffle=True, generator=generator,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    development_loader = DataLoader(
        development, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    return train, development, train_loader, development_loader


def forward(model, batch):
    return model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"], batch["instruction_embedding"],
    )


def evaluate(model, dataset, device, medians) -> dict:
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    yw, pw, yn, pn, truth_opv, pred_opv = [], [], [], [], [], []
    correct = 0; random_sum = 0.0; ranked_steps = 0
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move_batch(cpu, device)
            output = forward(model, batch)
            steps = int(batch["step_mask"].sum())
            for step in range(steps):
                mask = batch["candidate_mask"][0, step]
                if not bool(mask.any()):
                    continue
                tw = batch["option_cost"][0, step, mask]
                tn = batch["option_cost_without_checkpoint"][0, step, mask]
                pw_step = output.q_with_checkpoint[0, step, mask]
                pn_step = output.q_without_checkpoint[0, step, mask]
                yw.extend(tw.cpu().tolist()); pw.extend(pw_step.cpu().tolist())
                yn.extend(tn.cpu().tolist()); pn.extend(pn_step.cpu().tolist())
                truth_opv.append(float((tn - tw).max().cpu()))
                pred_opv.append(float((pn_step - pw_step).max().cpu()))
                if len(tw) >= 2 and float(tw.max() - tw.min()) > 1e-6:
                    correct += int(int(torch.argmin(tw)) == int(torch.argmin(pw_step)))
                    random_sum += 1.0 / len(tw); ranked_steps += 1
    yw = np.asarray(yw); pw = np.asarray(pw)
    yn = np.asarray(yn); pn = np.asarray(pn)
    truth_opv = np.asarray(truth_opv); pred_opv = np.asarray(pred_opv)
    labels = (truth_opv > 1e-6).astype(np.int64)
    return {
        "q_with_mae": float(np.mean(np.abs(yw - pw))),
        "q_with_train_median_baseline_mae": float(
            np.mean(np.abs(yw - medians[0]))
        ),
        "q_without_mae": float(np.mean(np.abs(yn - pn))),
        "q_without_train_median_baseline_mae": float(
            np.mean(np.abs(yn - medians[1]))
        ),
        "opv_mae": float(np.mean(np.abs(truth_opv - pred_opv))),
        "opv_zero_baseline_mae": float(np.mean(np.abs(truth_opv))),
        "opv_auc": rank_auc(labels, pred_opv),
        "best_option_accuracy": correct / ranked_steps,
        "best_option_random_accuracy": random_sum / ranked_steps,
        "ranked_steps": ranked_steps,
        "opv_positive_steps": int(labels.sum()),
        "evaluated_steps": len(truth_opv),
        "predicted_zero_opv_fraction": float(np.mean(pred_opv <= 1e-8)),
        "q_order_violations": int(np.sum(pw > pn + 1e-6)),
    }


def run(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("seed outside R3.3 protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3.3 protocol must be sealed")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    train, development, train_loader, development_loader = loaders(seed)
    source = R3_1 / f"augmented_seed_{seed}/relational_expiry_ree.pt"
    source_hash_before = sha256_file(source)
    model = CausalPairedQAdapter(768, 96, 128.0).to(device)
    objective = CausalPairedQAdapterLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = None; best_state = None; stale = 0; history = []
    for epoch in range(1, 21):
        model.train(); train_sum = 0.0; train_count = 0
        for cpu in train_loader:
            batch = move_batch(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(forward(model, batch), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite R3.3 loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval(); dev_sum = 0.0; dev_count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = move_batch(cpu, device)
                losses = objective(forward(model, batch), batch)
                size = int(batch["step_mask"].sum())
                dev_sum += float(losses["total"]) * size; dev_count += size
        native = dev_sum / dev_count
        history.append({"epoch": epoch, "train_total": train_sum / train_count,
                        "development_total": native})
        if best_loss is None or native < best_loss:
            best_loss = native
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state, strict=True)
    medians = scalar_train_medians(train)
    metrics = evaluate(model, development, device, medians)
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("frozen R3.1 checkpoint changed")
    checkpoint = run_dir / "causal_paired_q_opp.pt"
    torch.save({
        "schema_version": "revealnav-mf2-causal-opp-q-checkpoint/3.3",
        "seed": seed, "condition": "augmented", "hidden_dim": 96,
        "age_denominator": 128.0, "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "frozen_r3_1_checkpoint_sha256": source_hash_after,
        "train_medians": {"q_with": medians[0], "q_without": medians[1]},
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-causal-opp-q-run/3.3",
        "status": "CAUSAL_OPP_Q_R3_3_RUN_COMPLETE", "seed": seed,
        "metrics": metrics, "history": history,
        "train_events": len(train), "development_events": len(development),
        "source_r3_1_sha256_before": source_hash_before,
        "source_r3_1_sha256_after": source_hash_after,
        "source_r3_1_unchanged": source_hash_before == source_hash_after,
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                       "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_file(checkpoint)},
        "gold_payload_read": False, "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({"status": value["status"], "seed": seed,
                      "metrics": metrics}, indent=2))
    return 0


def summary(values):
    return {"mean": statistics.mean(values),
            "population_std": statistics.pstdev(values), "values": values}


def aggregate() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3.3 protocol drift")
    rows = [json.loads((OUT / f"seed_{seed}/result.json").read_text())
            for seed in SEEDS]
    if any(row.get("status") != "CAUSAL_OPP_Q_R3_3_RUN_COMPLETE" for row in rows):
        raise RuntimeError("incomplete R3.3 runs")
    float_names = [key for key, value in rows[0]["metrics"].items()
                   if isinstance(value, float)]
    results = {name: summary([row["metrics"][name] for row in rows])
               for name in float_names}
    gates = {
        "mean_q_with_mae_beats_train_median": (
            results["q_with_mae"]["mean"]
            < results["q_with_train_median_baseline_mae"]["mean"]
        ),
        "mean_q_without_mae_beats_train_median": (
            results["q_without_mae"]["mean"]
            < results["q_without_train_median_baseline_mae"]["mean"]
        ),
        "best_option_accuracy_above_random_in_two_seeds": sum(
            row["metrics"]["best_option_accuracy"]
            > row["metrics"]["best_option_random_accuracy"] for row in rows
        ) >= 2,
        "opv_mae_beats_zero_in_two_seeds": sum(
            row["metrics"]["opv_mae"]
            < row["metrics"]["opv_zero_baseline_mae"] for row in rows
        ) >= 2,
        "opv_auc_above_0_5_in_two_seeds": sum(
            row["metrics"]["opv_auc"] > 0.5 for row in rows
        ) >= 2,
        "q_order_invariant": all(
            row["metrics"]["q_order_violations"] == 0 for row in rows
        ),
        "source_r3_1_checkpoints_unchanged": all(
            row["source_r3_1_unchanged"] for row in rows
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-causal-opp-q-comparison/3.3",
        "status": "CAUSAL_OPP_Q_R3_3_GATE_PASS" if passed
                  else "CAUSAL_OPP_Q_R3_3_GATE_FAIL",
        "results": results, "gates": gates,
        "selected_seeds": list(SEEDS) if passed else [],
        "sources": {"protocol_sha256": sha256_file(PROTOCOL),
                    "manifest_sha256": sha256_file(MANIFEST)},
        "gold_payload_read": False, "paper_result": False,
        "next_step": "learned ECOG/OPP development evaluation" if passed else
                     "retain expiry-only result; Q representation blocked",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "results": results}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    if args.seed is None:
        parser.error("--seed required with --run")
    return run(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
