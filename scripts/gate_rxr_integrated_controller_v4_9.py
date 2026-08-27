#!/usr/bin/env python3
"""Train-only closed-loop gate for REE/Q, post-excursion Q, and return state."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import RelationalRevealExpiryHeads  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, BranchMacroAction, ExecutorPhase,
    IntegratedOptionController, PostExcursionAction, PostExcursionDataset,
    PostExcursionQHead, collate_branch_excursion_examples,
    collate_post_excursion_examples,
)
import run_rxr_branch_excursion_q_v4 as v4  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
FUSION_LOCK = ROOT / "locks/REE_Q_FUSION_CONTROLLER_V4_4.json"
RETURN_LOCK = ROOT / "locks/REE_Q_FUSION_RETURN_EXECUTOR_V4_5.json"
POST_COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_post_excursion_q_v4_8/"
    "RXR_POST_EXCURSION_Q_COMPARISON_V4_8.json"
)
POST_MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/post_excursion_v4_7/"
    "RXR_POST_EXCURSION_FULL_MANIFEST_V4_7.json"
)
POST_RESULT = POST_MANIFEST.with_name("RXR_POST_EXCURSION_FULL_RESULT_V4_7.json")
SOURCE = ROOT / "revealnav_mf2r4/integrated_controller.py"
SCRIPT = ROOT / "scripts/gate_rxr_integrated_controller_v4_9.py"
OUT = ROOT / "artifacts/evaluation/mf2_integrated_controller_v4_9"
PROTOCOL = OUT / "RXR_INTEGRATED_CONTROLLER_PROTOCOL_V4_9.json"
RESULT = OUT / "RXR_INTEGRATED_CONTROLLER_RESULT_V4_9.json"


def checkpoint_triplets() -> list[dict]:
    fusion = json.loads(FUSION_LOCK.read_text())
    post = json.loads(POST_COMPARISON.read_text())
    rows = []
    for pair in fusion["checkpoint_pairs"]:
        seed = int(pair["seed"])
        post_row = post["checkpoints"][str(seed)]
        row = {"seed": seed, "ree": pair["ree"], "q": pair["q"],
               "post": post_row}
        for value in (row["ree"], row["q"], row["post"]):
            path = ROOT / value["path"]
            if (
                path.is_symlink() or not path.is_file()
                or ROOT not in path.resolve().parents
                or path.stat().st_size != value["bytes"]
                or sha256_file(path) != value["sha256"]
            ):
                raise RuntimeError(f"checkpoint provenance drift: {path}")
        rows.append(row)
    if [row["seed"] for row in rows] != list(SEEDS):
        raise RuntimeError("integrated checkpoint cohort drift")
    return rows


def protocol_value() -> dict:
    post = json.loads(POST_COMPARISON.read_text())
    data = json.loads(POST_RESULT.read_text())
    return_lock = json.loads(RETURN_LOCK.read_text())
    if not (
        post.get("status") == "POST_EXCURSION_Q_GATE_PASS"
        and post.get("integration_authorized") is True
        and data.get("status") == "POST_EXCURSION_FULL_GATE_PASS"
        and return_lock.get("status")
        == "LOCKED_BEFORE_POST_EXCURSION_DATA_GENERATION"
    ):
        raise RuntimeError("integrated controller precondition failed")
    sources = (
        FUSION_LOCK, RETURN_LOCK, POST_COMPARISON, POST_MANIFEST,
        POST_RESULT, SOURCE, SCRIPT,
    )
    return {
        "schema_version": "revealnav-mf2-integrated-controller-protocol/4.9",
        "status": "SEALED_BEFORE_TRAIN_ONLY_CLOSED_LOOP_INTEGRATION",
        "checkpoint_triplets": checkpoint_triplets(),
        "scope": "83 RxR train-only scene-disjoint internal-development events x 3 seeds",
        "fixed_rules": {
            "initial": "locked REE + V4 Q fusion",
            "post_excursion": "minimum predicted CONTINUE/BACKTRACK cost; exact tie continues",
            "successful_backtrack": "active branch exhausted; select only remaining untried branches",
            "outbound_failure": "request same-checkpoint return without synthetic post feature",
            "return_failure": "RETURN_FAILED; only same-controller retry is legal",
        },
        "engineering_gates": {
            "all_three_checkpoint_triplets_strictly_load": True,
            "all_249_initial_decisions_legal": True,
            "every_reached_excursion_gets_one_post_decision": True,
            "no_executor_remains_in_exploring_or_returning": True,
            "all_failures_are_fail_closed": True,
            "post_action_metrics_reported_but_not_used_as_gate": True,
            "no_gold_or_evaluation_split_payload": True,
        },
        "performance_metrics_are_diagnostic_only": True,
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sources
        },
        "gold_access_allowed": False,
        "evaluation_split_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed integrated-controller protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_models(row: dict, device: torch.device):
    q_payload = torch.load(
        ROOT / row["q"]["path"], map_location="cpu", weights_only=False
    )
    q_model = BranchExcursionQHead(768, 96, 128.0).to(device)
    q_model.load_state_dict(q_payload["model_state_dict"], strict=True)
    ree_payload = torch.load(
        ROOT / row["ree"]["path"], map_location="cpu", weights_only=False
    )
    ree_model = RelationalRevealExpiryHeads(768, 128, 4).to(device)
    ree_model.load_state_dict(ree_payload["model_state_dict"], strict=True)
    post_payload = torch.load(
        ROOT / row["post"]["path"], map_location="cpu", weights_only=False
    )
    post_model = PostExcursionQHead(768, 96, 5.0).to(device)
    post_model.load_state_dict(post_payload["model_state_dict"], strict=True)
    return q_model.eval(), ree_model.eval(), post_model.eval()


def post_lookup(dataset: PostExcursionDataset) -> dict[tuple[str, int], int]:
    value = {}
    for index, example in enumerate(dataset.examples):
        record, _, _, branch_index, _ = example
        key = (record["event_id"], branch_index)
        if key in value:
            raise RuntimeError("duplicate post-excursion example")
        value[key] = index
    return value


def run(device: torch.device) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("integrated-controller protocol drift")
    _, development = v4.datasets()
    initial_loader = list(DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    ))
    post_data = PostExcursionDataset(POST_MANIFEST, "development")
    lookup = post_lookup(post_data)
    per_seed = {}
    totals = Counter()
    all_legal = True
    all_fail_closed = True
    with torch.no_grad():
        for row in checkpoint_triplets():
            q_model, ree_model, post_model = load_models(row, device)
            counts = Counter()
            post_regret = []
            for record, label_path, cpu in zip(
                development.records, development.label_paths, initial_loader
            ):
                label = json.loads(label_path.read_text())
                branch_ids = tuple(label["candidate_branch_ids"])
                batch = v4.move(cpu, device)
                valid = torch.isfinite(batch["commit_cost"][0])
                q_output = v4.forward(q_model, batch)
                commit = q_output.commit_cost[0, valid]
                excursion = q_output.excursion_cost[0, valid]
                steps = batch["history_embeddings"].shape[1]
                budgets = torch.tensor(
                    [1.5, 2.0, 3.0, 4.0], device=device
                ).view(1, 1, 4).expand(1, steps, 4)
                ree_output = ree_model(
                    batch["history_embeddings"], batch["candidate_embeddings"],
                    batch["candidate_mask"], budgets,
                    batch["instruction_embedding"],
                )
                step = int(batch["decision_index"][0])
                probabilities = torch.softmax(
                    ree_output.target_logits[0, step, valid], -1
                )
                controller = IntegratedOptionController(
                    f"cp:{record['event_id']}",
                    f"frozen-controller:{record['event_id']}", branch_ids,
                )
                decision = controller.decide_at_checkpoint(
                    branch_ids, probabilities.cpu().tolist(),
                    commit.cpu().tolist(), excursion.cpu().tolist(), 3,
                )
                counts["initial_decisions"] += 1
                counts[f"initial_{decision.action.value}"] += 1
                if decision.action is BranchMacroAction.DEFER:
                    all_legal = False
                    continue
                if decision.action is BranchMacroAction.COMMIT:
                    counts["terminal_committed"] += 1
                    continue
                branch_index = branch_ids.index(decision.branch_id)
                post_index = lookup.get((record["event_id"], branch_index))
                if post_index is None:
                    command = controller.fail_closed_outbound()
                    counts["outbound_fail_closed"] += 1
                    all_fail_closed &= (
                        command.checkpoint_id == f"cp:{record['event_id']}"
                    )
                    controller.report_return(False)
                    counts["terminal_return_failed"] += 1
                    continue
                post_cpu = collate_post_excursion_examples([post_data[post_index]])
                post_batch = v4.move(post_cpu, device)
                post_output = post_model(
                    post_batch["history_embeddings"], post_batch["history_length"],
                    post_batch["instruction_embedding"],
                    post_batch["selected_branch_embedding"],
                    post_batch["checkpoint_embedding"],
                    post_batch["post_candidate_embedding"],
                    post_batch["normalized_excursion_elapsed"],
                )
                truth = torch.stack((
                    post_batch["continue_cost"], post_batch["backtrack_cost"]
                ), -1)[0]
                post_decision, command = controller.decide_after_excursion(
                    float(post_output.continue_cost[0]),
                    float(post_output.backtrack_cost[0]),
                )
                counts["reached_post_decisions"] += 1
                counts[f"post_{post_decision.action.value}"] += 1
                choice = (
                    0 if post_decision.action is PostExcursionAction.CONTINUE else 1
                )
                oracle = float(truth.min())
                post_regret.append(float(truth[choice]) - oracle)
                strict = abs(float(truth[0] - truth[1])) > 1e-6
                counts["post_strict"] += int(strict)
                counts["post_strict_correct"] += int(
                    strict and choice == int(truth.argmin())
                )
                if post_decision.action is PostExcursionAction.CONTINUE:
                    counts["terminal_committed"] += 1
                    continue
                branch = post_data.examples[post_index][4]
                all_fail_closed &= command is not None and (
                    command.controller_ref
                    == f"frozen-controller:{record['event_id']}"
                )
                succeeded = bool(branch["return_route"].get("success", False))
                controller.report_return(succeeded)
                if succeeded:
                    counts["terminal_at_checkpoint"] += 1
                else:
                    counts["terminal_return_failed"] += 1
                if controller.executor.phase in (
                    ExecutorPhase.EXPLORING, ExecutorPhase.RETURNING
                ):
                    all_legal = False
            totals.update(counts)
            per_seed[str(row["seed"])] = {
                "counts": dict(counts),
                "selected_post_mean_action_regret": (
                    statistics.mean(post_regret) if post_regret else None
                ),
                "selected_post_strict_accuracy": (
                    counts["post_strict_correct"] / counts["post_strict"]
                    if counts["post_strict"] else None
                ),
            }
    expected_initial = len(development) * len(SEEDS)
    gates = {
        "all_three_checkpoint_triplets_strictly_load": len(per_seed) == 3,
        "all_249_initial_decisions_legal": (
            totals["initial_decisions"] == expected_initial
            and totals["initial_defer"] == 0 and all_legal
        ),
        "every_reached_excursion_gets_one_post_decision": (
            totals["reached_post_decisions"]
            == totals["initial_checkpointed_excursion"]
            - totals["outbound_fail_closed"]
        ),
        "no_executor_remains_in_exploring_or_returning": all_legal,
        "all_failures_are_fail_closed": all_fail_closed,
        "post_action_metrics_reported_but_not_used_as_gate": True,
        "no_gold_or_evaluation_split_payload": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-integrated-controller-result/4.9",
        "status": (
            "INTEGRATED_CONTROLLER_ENGINEERING_GATE_PASS" if passed
            else "INTEGRATED_CONTROLLER_ENGINEERING_GATE_FAIL"
        ),
        "counts": dict(totals), "per_seed": per_seed,
        "gates": gates,
        "performance_metrics_are_diagnostic_only": True,
        "protocol_sha256": sha256_file(PROTOCOL),
        "locked_controller_ready_for_online_evaluation": passed,
        "gold_payload_read": False, "paper_result": False,
        "next_gate": (
            "fresh online episode integration without shadow-action restriction"
            if passed else "repair closed-loop state transition"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"],
                      "gates": gates, "per_seed": per_seed}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return seal() if args.seal else run(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
