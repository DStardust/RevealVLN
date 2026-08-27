#!/usr/bin/env python3
"""Numerical adjudication of candidate permutation equivariance for V4 Q."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import collate_branch_excursion_examples  # noqa: E402
import adjudicate_rxr_branch_excursion_q_ties_v4_1 as ties  # noqa: E402
import run_rxr_branch_excursion_q_v4 as training  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


FAILED = ties.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4_2"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_NUMERIC_PROTOCOL_V4_2.json"
RESULT = OUT / "RXR_BRANCH_EXCURSION_Q_NUMERIC_RESULT_V4_2.json"


def protocol_value() -> dict:
    failed = json.loads(FAILED.read_text())
    false_gates = sorted(key for key, passed in failed["gates"].items() if not passed)
    if not (
        failed.get("status") == "BRANCH_EXCURSION_Q_TIE_AWARE_ENGINEERING_FAIL"
        and false_gates == ["checkpoint_candidate_permutation_equivariance"]
        and failed.get("gold_payload_read") is False
    ):
        raise RuntimeError("numeric adjudication precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-q-numeric/4.2",
        "status": "SEALED_AFTER_V4_1_NUMERIC_FAILURE_BEFORE_ADJUDICATION",
        "observed_gpu_float32_max_abs": failed["aggregate"][
            "candidate_permutation_max_abs"
        ],
        "tests": {
            "gpu_float32_max_abs_at_most_1e_5": True,
            "gpu_float32_argmin_exact": True,
            "cpu_float64_max_abs_at_most_1e_10": True,
            "cpu_float64_argmin_exact": True,
        },
        "interpretation": (
            "Candidate-set reduction order may change float32 accumulation at a "
            "few 1e-6. Equivariance requires exact discrete decisions and a "
            "float64 structural witness; V4.1 remains failed and is not rewritten."
        ),
        "sources": {
            str(FAILED.relative_to(ROOT)): sha256_file(FAILED),
            str(ties.PROTOCOL.relative_to(ROOT)): sha256_file(ties.PROTOCOL),
            **{
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in ties.checkpoint_paths().values()
            },
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed numeric protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def dtype_batch(batch, device, dtype):
    return {
        key: value.to(device=device, dtype=dtype)
        if value.is_floating_point() else value.to(device)
        for key, value in batch.items()
    }


def evaluate(seed: int, device, dtype) -> dict:
    model, _ = ties.load_model(seed, device)
    model = model.to(dtype=dtype)
    _, development = training.datasets()
    loader = DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    maximum = 0.0
    exact = 0
    events = 0
    with torch.no_grad():
        for cpu in loader:
            batch = dtype_batch(cpu, device, dtype)
            output = ties.forward(model, batch)
            count = batch["candidate_embeddings"].shape[2]
            order = torch.arange(count - 1, -1, -1, device=device)
            permuted = dict(batch)
            permuted["candidate_embeddings"] = batch["candidate_embeddings"][:, :, order]
            permuted["candidate_mask"] = batch["candidate_mask"][:, :, order]
            changed = ties.forward(model, permuted)
            restored_commit = changed.commit_cost[:, order]
            restored_excursion = changed.excursion_cost[:, order]
            valid = torch.isfinite(output.commit_cost)
            maximum = max(
                maximum,
                float((restored_commit[valid] - output.commit_cost[valid]).abs().max()),
                float((restored_excursion[valid] - output.excursion_cost[valid]).abs().max()),
            )
            original_action = torch.cat((
                output.commit_cost[0, valid[0]], output.excursion_cost[0, valid[0]]
            )).argmin()
            restored_action = torch.cat((
                restored_commit[0, valid[0]], restored_excursion[0, valid[0]]
            )).argmin()
            exact += int(original_action == restored_action)
            events += 1
    return {
        "maximum_cost_abs": maximum,
        "argmin_agreement": exact / events,
        "events": events,
    }


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("numeric protocol must be sealed")
    per_seed = {}
    for seed in training.SEEDS:
        per_seed[str(seed)] = {
            "gpu_float32": evaluate(seed, torch.device("cuda"), torch.float32),
            "cpu_float64": evaluate(seed, torch.device("cpu"), torch.float64),
        }
    gpu_max = max(row["gpu_float32"]["maximum_cost_abs"] for row in per_seed.values())
    cpu_max = max(row["cpu_float64"]["maximum_cost_abs"] for row in per_seed.values())
    gpu_agreement = statistics.mean(
        row["gpu_float32"]["argmin_agreement"] for row in per_seed.values()
    )
    cpu_agreement = statistics.mean(
        row["cpu_float64"]["argmin_agreement"] for row in per_seed.values()
    )
    gates = {
        "gpu_float32_max_abs_at_most_1e_5": gpu_max <= 1e-5,
        "gpu_float32_argmin_exact": gpu_agreement == 1.0,
        "cpu_float64_max_abs_at_most_1e_10": cpu_max <= 1e-10,
        "cpu_float64_argmin_exact": cpu_agreement == 1.0,
        "v4_1_failure_preserved": json.loads(FAILED.read_text())["status"]
        == "BRANCH_EXCURSION_Q_TIE_AWARE_ENGINEERING_FAIL",
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-numeric-result/4.2",
        "status": (
            "BRANCH_EXCURSION_Q_NUMERICAL_ADJUDICATION_PASS" if passed
            else "BRANCH_EXCURSION_Q_NUMERICAL_ADJUDICATION_FAIL"
        ),
        "per_seed": per_seed,
        "aggregate": {
            "gpu_float32_max_abs": gpu_max,
            "gpu_float32_argmin_agreement": gpu_agreement,
            "cpu_float64_max_abs": cpu_max,
            "cpu_float64_argmin_agreement": cpu_agreement,
        },
        "gates": gates,
        "protocol_sha256": sha256_file(PROTOCOL),
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "unseen controller integration" if passed else "numeric diagnosis",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "aggregate": value["aggregate"],
        "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    return seal() if args.seal else run()


if __name__ == "__main__":
    raise SystemExit(main())
