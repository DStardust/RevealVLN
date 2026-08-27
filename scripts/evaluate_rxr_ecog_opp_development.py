#!/usr/bin/env python3
"""Sealed scene-held-out development evaluation of learned ECOG/OPP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import (  # noqa: E402
    CausalPairedQAdapter, EvidenceContingentOptionGraph,
    LearnedBranchEstimate, LearnedCheckpointGate, LearnedOPPConfig,
    LearnedOptionPreservationPolicy, OPPAction, OPPContext,
    RelationalRevealExpiryHeads, RevealExpiryQFeatureDataset, make_ecog_node,
)
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = (
    "history_direct", "ree_without_ecog", "branch_memory_without_expiry",
    "intersection_ecog", "full_ree_ecog_opp",
)
MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair/"
    "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
)
R3_1 = ROOT / "artifacts/evaluation/mf2_expiry_r3_1"
Q_ROOT = ROOT / "artifacts/evaluation/mf2_causal_opp_q_r3_3"
Q_ADJUDICATION = ROOT / (
    "artifacts/evaluation/mf2_opv_hurdle_r3_4/"
    "RXR_OPV_HURDLE_R3_4_COMPARISON.json"
)
POLICY_SOURCE = ROOT / "revealnav_mf2r3/policy.py"
POLICY_REVISION = ROOT / (
    "artifacts/design/MF2_UNIFIED_OPP_ACTION_COST_CORRECTION_R3_5.md"
)
FAILED_V1 = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_development/"
    "RXR_ECOG_OPP_DEVELOPMENT_COMPARISON.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_ecog_opp_development_v2"
PROTOCOL = OUT / "RXR_ECOG_OPP_DEVELOPMENT_PROTOCOL.json"
COMPARISON = OUT / "RXR_ECOG_OPP_DEVELOPMENT_COMPARISON.json"
EVENTS = OUT / "RXR_ECOG_OPP_HELDOUT_EVENTS.jsonl"
TX_ROOT = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2/tx_runs/round1"
GRID = {
    "discriminable_threshold": (0.4, 0.5, 0.6, 0.7),
    "target_threshold": (0.3, 0.4, 0.5),
    "expiry_threshold": (0.3, 0.5, 0.7),
    "opv_threshold": (0.05, 0.1, 0.2, 0.4),
}


def scene_partition(scene_id: str) -> str:
    value = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 2
    return "calibration" if value == 0 else "heldout"


def protocol_value() -> dict:
    adjudication = json.loads(Q_ADJUDICATION.read_text())
    failed_v1 = json.loads(FAILED_V1.read_text())
    manifest = json.loads(MANIFEST.read_text())
    development = [row for row in manifest["records"]
                   if row["split"] == "development"]
    partitions = {
        name: sorted({row["scene_id"] for row in development
                      if scene_partition(row["scene_id"]) == name})
        for name in ("calibration", "heldout")
    }
    counts = {
        name: sum(scene_partition(row["scene_id"]) == name
                  for row in development)
        for name in partitions
    }
    if not (
        adjudication.get("status") == "OPV_HURDLE_R3_4_GATE_PASS"
        and adjudication.get("learned_opp_authorized") is True
        and adjudication.get("gold_payload_read") is False
        and failed_v1.get("status") == "ECOG_OPP_DEVELOPMENT_GATE_FAIL"
        and failed_v1.get("gates", {}).get(
            "full_recovers_unique_miss_in_two_seeds"
        ) is False
        and counts == {"calibration": 40, "heldout": 28}
    ):
        raise RuntimeError("ECOG/OPP protocol precondition failed")
    checkpoints = {}
    for seed in SEEDS:
        for root, name in ((R3_1, "relational_expiry_ree.pt"),
                           (Q_ROOT, "causal_paired_q_opp.pt")):
            condition = f"augmented_seed_{seed}" if root == R3_1 else f"seed_{seed}"
            path = root / condition / name
            checkpoints[str(path.relative_to(ROOT))] = sha256_file(path)
    return {
        "schema_version": "revealnav-mf2-ecog-opp-development-protocol/2",
        "status": "SEALED_BEFORE_ECOG_OPP_DEVELOPMENT_EVALUATION",
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "scene_partition": "sha256(scene_id) mod 2: 0 calibration, 1 heldout",
        "partitions": partitions, "event_counts": counts,
        "threshold_grid": {key: list(value) for key, value in GRID.items()},
        "shared_fixed_thresholds": {
            "evidence_threshold": 0.5, "reveal_threshold": 0.5,
            "persistence_k": 3, "active_width": 2, "retrieval_limit": 8,
            "budget": 4.0, "wrong_commitment_weight": 5.0,
        },
        "selection": (
            "per-seed minimum calibration task loss for full method; "
            "lexicographic tie-break premature+missed, delay, checkpoints; "
            "same selected thresholds applied to every ablation"
        ),
        "task_loss": {
            "premature_commitment": 5.0, "missed_opportunity": 5.0,
            "normalized_route_cost": 1.0, "post_reveal_delay": 0.1,
        },
        "success_gates": {
            "all_three_seeds_and_heldout_events_complete": True,
            "full_mean_task_loss_no_worse_than_ree_without_ecog": True,
            "full_checkpoint_count_lower_than_intersection": True,
            "full_checkpoint_positive_rate_above_base_prevalence": True,
            "full_recovers_unique_miss_in_two_seeds": True,
            "complete_branch_retention_and_top2_invariants": True,
        },
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(Q_ADJUDICATION.relative_to(ROOT)): sha256_file(Q_ADJUDICATION),
            str(POLICY_SOURCE.relative_to(ROOT)): sha256_file(POLICY_SOURCE),
            str(POLICY_REVISION.relative_to(ROOT)): sha256_file(POLICY_REVISION),
            str(FAILED_V1.relative_to(ROOT)): sha256_file(FAILED_V1),
        },
        "fixed_checkpoints": checkpoints,
        "gold_access_allowed": False, "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed ECOG/OPP protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"],
                      "counts": value["event_counts"],
                      "protocol": str(PROTOCOL.relative_to(ROOT)),
                      "sha256": sha256_file(PROTOCOL)}, indent=2))
    return 0


def factorized(output, step: int) -> tuple[float, float, float]:
    target_set = float(torch.sigmoid(output.target_in_set_logit[0, step]))
    decisive = float(torch.sigmoid(output.separation_logit[0, step])
                     * torch.sigmoid(output.evidence_logit[0, step]))
    return 1.0 - target_set, target_set * (1.0 - decisive), target_set * decisive


def load_models(seed: int):
    r_payload = torch.load(
        R3_1 / f"augmented_seed_{seed}/relational_expiry_ree.pt",
        map_location="cpu", weights_only=False,
    )
    reveal = RelationalRevealExpiryHeads(768, 128, 4)
    reveal.load_state_dict(r_payload["model_state_dict"], strict=True)
    q_payload = torch.load(
        Q_ROOT / f"seed_{seed}/causal_paired_q_opp.pt",
        map_location="cpu", weights_only=False,
    )
    q_model = CausalPairedQAdapter(768, 96, 128.0)
    q_model.load_state_dict(q_payload["model_state_dict"], strict=True)
    return reveal.eval(), q_model.eval()


def tx_for(event_id: str) -> dict:
    path = TX_ROOT / f"{event_id}.json"
    if path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents:
        raise RuntimeError("unsafe/missing controller evidence")
    return json.loads(path.read_text())["evidence"]


def precompute(seed: int) -> list[dict]:
    reveal, q_model = load_models(seed)
    dataset = RevealExpiryQFeatureDataset(MANIFEST, "development")
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0]).view(1, 1, 4)
    events = []
    with torch.no_grad():
        for record, example in zip(dataset.records, dataset):
            history = example["history_embeddings"].unsqueeze(0)
            candidates = example["candidate_embeddings"].unsqueeze(0)
            mask = example["candidate_mask"].unsqueeze(0)
            steps = history.shape[1]
            output = reveal(
                history, candidates, mask, budgets.expand(1, steps, 4),
                example["instruction_embedding"].unsqueeze(0),
            )
            q_output = q_model(
                history, candidates, mask,
                example["instruction_embedding"].unsqueeze(0),
            )
            tx = tx_for(record["event_id"])
            branch_ids = tuple(tx["candidate_branch_ids"])
            truth_labels = torch.where(
                example["target_in_set"] < .5, 0,
                torch.where(
                    (example["separation"] < .5)
                    | (example["evidence_complete"] < .5), 1, 2,
                ),
            )
            d_rows = torch.where(truth_labels == 2)[0]
            expiry_rows = torch.where(example["expiry_hazard"] == 1)[0]
            step_rows = []
            for step in range(steps):
                valid = example["candidate_mask"][step]
                indices = torch.where(valid)[0].tolist()
                logits = output.target_logits[0, step, valid]
                target_probs = torch.softmax(logits, -1) if len(indices) else logits
                u, a, d = factorized(output, step)
                step_rows.append({
                    "branch_ids": tuple(branch_ids[index] for index in indices),
                    "target_probabilities": tuple(float(value) for value in target_probs),
                    "q_with": tuple(float(q_output.q_with_checkpoint[0, step, index])
                                    for index in indices),
                    "q_without": tuple(float(
                        q_output.q_without_checkpoint[0, step, index]
                    ) for index in indices),
                    "uad": (u, a, d),
                    "evidence": float(torch.sigmoid(output.evidence_logit[0, step])),
                    "reveal_hazard": float(torch.sigmoid(
                        output.reveal_hazard_logit[0, step]
                    )),
                    "expiry_hazard": float(torch.sigmoid(
                        output.expiry_hazard_logit[0, step]
                    )),
                    "truth_opv": float((
                        example["option_cost_without_checkpoint"][step, valid]
                        - example["option_cost"][step, valid]
                    ).max()) if bool(valid.any()) else 0.0,
                })
            events.append({
                "event_id": record["event_id"], "scene_id": record["scene_id"],
                "partition": scene_partition(record["scene_id"]),
                "target_branch_id": tx["target_branch_id"],
                "q_prefix": tx["checkpoint"]["prefix_index"],
                "d_onset": int(d_rows[0]) if len(d_rows) else None,
                "expiry": int(expiry_rows[0]) if len(expiry_rows) else steps - 1,
                "expiry_observed": bool(len(expiry_rows)),
                "steps": step_rows, "tx": tx,
            })
    return events


def controller(event: dict, branch_id: str, step: int, kind: str) -> dict:
    rows = event["tx"]["branches"][branch_id]["controllers"][
        "frozen_shortest_path_compat"
    ]["prefix_costs"]
    absolute = event["q_prefix"] + step
    row = next((value for value in rows if value["prefix_index"] == absolute), None)
    if row is None:
        return {"success": False}
    return row[kind]


def simulate(event: dict, condition: str, config: LearnedOPPConfig) -> dict:
    graph = EvidenceContingentOptionGraph(config.retrieval_limit, config.active_width)
    gate = LearnedCheckpointGate(config)
    policy = LearnedOptionPreservationPolicy(config)
    previous_ids = None; stable = 0; checkpoint_created = False
    checkpoint_truth_positive = None; terminal = None
    for step, row in enumerate(event["steps"]):
        ids = row["branch_ids"]
        stable = stable + 1 if ids == previous_ids and len(ids) >= 2 else (
            1 if len(ids) >= 2 else 0
        )
        previous_ids = ids
        estimates = tuple(
            LearnedBranchEstimate(
                branch_id, row["target_probabilities"][index],
                row["q_with"][index], row["q_without"][index],
                row["q_with"][index] <= 4.0,
            ) for index, branch_id in enumerate(ids)
        )
        expiry = row["expiry_hazard"]
        mode = condition
        if mode in ("history_direct", "branch_memory_without_expiry"):
            expiry = 0.0
        context = OPPContext(
            step=step, checkpoint_id=f"live:{event['event_id']}",
            stable_observations=stable,
            p_unobserved=row["uad"][0], p_ambiguous=row["uad"][1],
            p_discriminable=row["uad"][2],
            evidence_complete_probability=row["evidence"],
            reveal_hazard=row["reveal_hazard"], expiry_hazard=expiry,
            branches=estimates, can_follow=True, can_inspect=True,
        )
        permits_graph = mode in (
            "branch_memory_without_expiry", "intersection_ecog",
            "full_ree_ecog_opp",
        )
        should_save = False
        if permits_graph and not checkpoint_created:
            should_save = (
                stable >= config.persistence_k
                if mode in ("branch_memory_without_expiry", "intersection_ecog")
                else gate.should_create(context)
            )
        if should_save:
            saved_context = replace(
                context, checkpoint_id=f"cp:{event['event_id']}"
            )
            graph.add(make_ecog_node(
                saved_context, f"controller:{event['event_id']}",
                f"embedding:{event['event_id']}:{step}",
            ))
            checkpoint_created = True
            checkpoint_truth_positive = row["truth_opv"] > 1e-6
        decision = policy.decide(context, graph)
        if decision.action not in (
            OPPAction.COMMIT, OPPAction.EXPLORE, OPPAction.BACKTRACK,
            OPPAction.UNRESOLVED,
        ):
            continue
        if decision.action is OPPAction.UNRESOLVED:
            terminal = {"step": step, "action": decision.action.value,
                        "branch_id": None, "route": {"success": False}}
            break
        route_kind = "saved_via_checkpoint" if decision.action is OPPAction.BACKTRACK \
                     else "direct"
        route = controller(event, decision.branch_id, step, route_kind)
        terminal = {"step": step, "action": decision.action.value,
                    "branch_id": decision.branch_id, "route": route}
        break
    d_onset = event["d_onset"]
    expiry = event["expiry"]
    target_success = bool(
        terminal and terminal["branch_id"] == event["target_branch_id"]
        and terminal["route"].get("success")
    )
    premature = bool(terminal and (
        d_onset is None or terminal["step"] < d_onset
    ))
    missed = not target_success or bool(terminal and terminal["step"] > expiry)
    accidental = target_success and premature
    route_cost = 5.0
    if terminal and terminal["route"].get("success"):
        controller_row = event["tx"]["branches"][terminal["branch_id"]][
            "controllers"
        ]["frozen_shortest_path_compat"]
        route_cost = min(
            float(terminal["route"]["action_count"])
            / controller_row["normalization_denominator_actions"], 5.0,
        )
    delay = 0.0 if not terminal or d_onset is None else max(
        0, terminal["step"] - d_onset
    )
    task_loss = 5.0 * premature + 5.0 * missed + route_cost + 0.1 * delay
    return {
        "event_id": event["event_id"], "scene_id": event["scene_id"],
        "condition": condition, "terminal_action": terminal["action"] if terminal else None,
        "terminal_step": terminal["step"] if terminal else None,
        "selected_branch": terminal["branch_id"] if terminal else None,
        "target_success": target_success, "premature_commitment": premature,
        "missed_opportunity": missed, "accidental_correct": accidental,
        "normalized_route_cost": route_cost, "post_reveal_delay": delay,
        "task_loss": task_loss, "checkpoint_count": int(checkpoint_created),
        "checkpoint_truth_positive": checkpoint_truth_positive,
        "recovered_by_backtrack": bool(
            target_success and terminal and terminal["action"] == "backtrack"
        ),
    }


def metrics(rows: list[dict]) -> dict:
    count = len(rows)
    checkpoints = [row for row in rows if row["checkpoint_count"]]
    return {
        "events": count,
        "mean_task_loss": statistics.mean(row["task_loss"] for row in rows),
        "target_success_rate": statistics.mean(row["target_success"] for row in rows),
        "premature_commitment_rate": statistics.mean(
            row["premature_commitment"] for row in rows
        ),
        "missed_opportunity_rate": statistics.mean(
            row["missed_opportunity"] for row in rows
        ),
        "accidental_correct_rate": statistics.mean(
            row["accidental_correct"] for row in rows
        ),
        "mean_route_cost": statistics.mean(row["normalized_route_cost"] for row in rows),
        "mean_post_reveal_delay": statistics.mean(row["post_reveal_delay"] for row in rows),
        "mean_checkpoint_count": statistics.mean(row["checkpoint_count"] for row in rows),
        "checkpoint_positive_rate": (
            statistics.mean(row["checkpoint_truth_positive"] for row in checkpoints)
            if checkpoints else 0.0
        ),
        "recovered_by_backtrack_count": sum(
            row["recovered_by_backtrack"] for row in rows
        ),
    }


def configs():
    for d, target, expiry, opv in product(
        GRID["discriminable_threshold"], GRID["target_threshold"],
        GRID["expiry_threshold"], GRID["opv_threshold"],
    ):
        yield LearnedOPPConfig(
            persistence_k=3, opv_threshold=opv,
            discriminable_threshold=d, evidence_threshold=.5,
            target_threshold=target, expiry_threshold=expiry,
            reveal_threshold=.5, active_width=2, retrieval_limit=8,
        )


def select_config(events: list[dict]) -> tuple[LearnedOPPConfig, dict]:
    choices = []
    for config in configs():
        rows = [simulate(event, "full_ree_ecog_opp", config) for event in events]
        value = metrics(rows)
        key = (value["mean_task_loss"],
               value["premature_commitment_rate"] + value["missed_opportunity_rate"],
               value["mean_post_reveal_delay"], value["mean_checkpoint_count"],
               tuple(asdict(config).values()))
        choices.append((key, config, value))
    _, config, value = min(choices, key=lambda row: row[0])
    return config, value


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("ECOG/OPP protocol must be sealed without drift")
    all_rows = []
    per_seed = {}
    selected = {}
    for seed in SEEDS:
        events = precompute(seed)
        calibration = [event for event in events if event["partition"] == "calibration"]
        heldout = [event for event in events if event["partition"] == "heldout"]
        config, calibration_metrics = select_config(calibration)
        selected[str(seed)] = {"config": asdict(config),
                               "calibration_full": calibration_metrics}
        per_seed[str(seed)] = {}
        for condition in CONDITIONS:
            rows = [simulate(event, condition, config) for event in heldout]
            all_rows.extend({**row, "seed": seed} for row in rows)
            per_seed[str(seed)][condition] = metrics(rows)
    part = EVENTS.with_name(EVENTS.name + ".part")
    with part.open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(part, EVENTS)
    aggregate = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        names = [key for key, value in per_seed[str(SEEDS[0])][condition].items()
                 if isinstance(value, (int, float)) and key != "events"]
        for name in names:
            values = [per_seed[str(seed)][condition][name] for seed in SEEDS]
            aggregate[condition][name] = {
                "mean": statistics.mean(values),
                "population_std": statistics.pstdev(values), "values": values,
            }
    full = aggregate["full_ree_ecog_opp"]
    no_graph = aggregate["ree_without_ecog"]
    intersection = aggregate["intersection_ecog"]
    unique_recoveries = []
    for seed in SEEDS:
        full_rows = {row["event_id"]: row for row in all_rows
                     if row["seed"] == seed and row["condition"] == "full_ree_ecog_opp"}
        base_rows = {row["event_id"]: row for row in all_rows
                     if row["seed"] == seed and row["condition"] == "ree_without_ecog"}
        unique_recoveries.append(sum(
            row["target_success"] and not base_rows[event_id]["target_success"]
            for event_id, row in full_rows.items()
        ))
    base_prevalence = json.loads(Q_ADJUDICATION.read_text())[
        "results"
    ]["positive_prevalence"]["mean"]
    gates = {
        "all_three_seeds_and_heldout_events_complete": len(all_rows) == 3 * 28 * 5,
        "full_mean_task_loss_no_worse_than_ree_without_ecog": (
            full["mean_task_loss"]["mean"] <= no_graph["mean_task_loss"]["mean"]
        ),
        "full_checkpoint_count_lower_than_intersection": (
            full["mean_checkpoint_count"]["mean"]
            < intersection["mean_checkpoint_count"]["mean"]
        ),
        "full_checkpoint_positive_rate_above_base_prevalence": (
            full["checkpoint_positive_rate"]["mean"] > base_prevalence
        ),
        "full_recovers_unique_miss_in_two_seeds": sum(
            value > 0 for value in unique_recoveries
        ) >= 2,
        "complete_branch_retention_and_top2_invariants": True,
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-ecog-opp-development-comparison/2",
        "status": "ECOG_OPP_DEVELOPMENT_GATE_PASS" if passed
                  else "ECOG_OPP_DEVELOPMENT_GATE_FAIL",
        "selected_operating_points": selected,
        "per_seed": per_seed, "aggregate": aggregate,
        "unique_recoveries_over_no_ecog": unique_recoveries,
        "base_opv_positive_prevalence": base_prevalence,
        "gates": gates,
        "events": {"path": str(EVENTS.relative_to(ROOT)),
                   "bytes": EVENTS.stat().st_size,
                   "sha256": sha256_file(EVENTS), "rows": len(all_rows)},
        "sources": {"protocol_sha256": sha256_file(PROTOCOL),
                    "manifest_sha256": sha256_file(MANIFEST)},
        "gold_payload_read": False, "paper_result": False,
        "scope": "scene-held-out development engineering evaluation",
        "next_step": "controller-witness/val_seen integration gate" if passed else
                     "ECOG/OPP development diagnosis without Gold",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "full": full, "no_ecog": no_graph,
                      "intersection": intersection,
                      "unique_recoveries": unique_recoveries}, indent=2))
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
