#!/usr/bin/env python3
"""Fail-closed independent audit of the immutable MF3ZU RxR result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_evidence_memory_metrics import (  # noqa: E402
    ARM_CURRENT,
    ARM_MEMORY,
    ARM_SHUFFLED,
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SUBGROUPS,
    apply_fixed_rxr_gates,
    evaluate_three_arm_probe,
)
from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    EXPECTED_POPULATION_EPISODES,
    EXPECTED_POPULATION_ROWS,
    EXPECTED_POPULATION_SCENES,
    FOLDS,
    PUBLIC_CLOSED,
    RESULT_PATH,
    REVISION,
    ProtocolError,
    scene_fold_mapping,
    sha256_file,
    verify_protocol,
)
from scripts.train_mf3zu_rxr_feasibility import (  # noqa: E402
    MF3ZUTrainingError,
    ProbeArrays,
    _canonical_oof_jsonl,
    load_frozen_probe_inputs,
)


AUDIT_PATH = RESULT_PATH.with_name("MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT.json")


class MF3ZUResultAuditError(RuntimeError):
    """Raised when result evidence is incomplete or contradicts the seal."""


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MF3ZUResultAuditError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MF3ZUResultAuditError(f"malformed OOF row {number}")
        rows.append(value)
    return rows


def _recompute_oof_evaluation(
    oof_rows: Sequence[Mapping[str, object]],
    *,
    enforce_frozen_counts: bool,
    bootstrap_replicates: int,
) -> tuple[dict[str, object], dict[str, int]]:
    """Rebuild every reported scientific quantity from immutable OOF rows."""

    if not oof_rows:
        raise MF3ZUResultAuditError("OOF prediction file is empty")
    if enforce_frozen_counts and int(bootstrap_replicates) != BOOTSTRAP_REPLICATES:
        raise MF3ZUResultAuditError("formal audit requires 10000 bootstrap replicates")
    maximum_candidates = 0
    for row in oof_rows:
        ids = row.get("candidate_action_ids")
        if not isinstance(ids, list):
            raise MF3ZUResultAuditError("OOF candidate identities are missing")
        maximum_candidates = max(maximum_candidates, len(ids))
    if maximum_candidates < 2:
        raise MF3ZUResultAuditError("OOF has no rankable candidate set")

    row_count = len(oof_rows)
    target = np.full(row_count, -1, dtype=np.int64)
    mask = np.zeros((row_count, maximum_candidates), dtype=bool)
    scores = {
        arm: np.zeros((row_count, maximum_candidates), dtype=np.float64)
        for arm in ARMS
    }
    scenes: list[str] = []
    episodes: list[str] = []
    events: list[str] = []
    folds: list[int] = []
    required = np.zeros(row_count, dtype=bool)
    scene_to_folds: dict[str, set[int]] = {}
    episode_to_identity: dict[str, set[tuple[str, int]]] = {}
    for index, row in enumerate(oof_rows):
        event_id = str(row.get("event_id", ""))
        scene_id = str(row.get("scene_id", ""))
        episode_id = str(row.get("episode_id", ""))
        decision_step = row.get("decision_step")
        scene_fold = row.get("scene_fold")
        memory_required = row.get("memory_required")
        ids = row.get("candidate_action_ids")
        row_scores = row.get("scores")
        target_index = row.get("target_index")
        if not event_id or not scene_id or not episode_id:
            raise MF3ZUResultAuditError("OOF causal identity is incomplete")
        if isinstance(decision_step, bool) or not isinstance(decision_step, int) or decision_step < 0:
            raise MF3ZUResultAuditError("OOF decision step is invalid")
        if isinstance(scene_fold, bool) or not isinstance(scene_fold, int) or scene_fold not in range(FOLDS):
            raise MF3ZUResultAuditError("OOF scene fold is invalid")
        if type(memory_required) is not bool:
            raise MF3ZUResultAuditError("OOF memory-required label is not boolean")
        if (
            not isinstance(ids, list)
            or len(ids) < 2
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(row_scores, Mapping)
            or set(row_scores) != set(ARMS)
            or isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < len(ids)
        ):
            raise MF3ZUResultAuditError("OOF candidate schema is invalid")
        for arm in ARMS:
            values = row_scores[arm]
            if (
                not isinstance(values, list)
                or len(values) != len(ids)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in values
                )
            ):
                raise MF3ZUResultAuditError("OOF arm score schema is invalid")
            numeric = np.asarray(values, dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise MF3ZUResultAuditError("OOF arm score is non-finite")
            scores[arm][index, : len(ids)] = numeric
        events.append(event_id)
        scenes.append(scene_id)
        episodes.append(episode_id)
        folds.append(scene_fold)
        required[index] = memory_required
        target[index] = target_index
        mask[index, : len(ids)] = True
        scene_to_folds.setdefault(scene_id, set()).add(scene_fold)
        episode_to_identity.setdefault(episode_id, set()).add((scene_id, scene_fold))

    if len(set(events)) != row_count:
        raise MF3ZUResultAuditError("OOF event identity is repeated")
    if any(len(values) != 1 for values in scene_to_folds.values()):
        raise MF3ZUResultAuditError("one raw scene crosses OOF folds")
    if any(len(values) != 1 for values in episode_to_identity.values()):
        raise MF3ZUResultAuditError("one episode crosses scene/fold identity")
    if set(folds) != set(range(FOLDS)):
        raise MF3ZUResultAuditError("OOF does not cover all five folds")
    expected_scene_folds = scene_fold_mapping(scenes)
    if any(
        fold != expected_scene_folds[scene]
        for scene, fold in zip(scenes, folds, strict=True)
    ):
        raise MF3ZUResultAuditError("OOF fold differs from the sealed scene mapping")

    summary = {
        "rows": row_count,
        "episodes": len(set(episodes)),
        "raw_scenes": len(set(scenes)),
        "folds": len(set(folds)),
    }
    if enforce_frozen_counts and summary != {
        "rows": EXPECTED_POPULATION_ROWS,
        "episodes": EXPECTED_POPULATION_EPISODES,
        "raw_scenes": EXPECTED_POPULATION_SCENES,
        "folds": FOLDS,
    }:
        raise MF3ZUResultAuditError(f"formal OOF support count drift: {summary}")
    evaluation = evaluate_three_arm_probe(
        scores,
        target,
        mask,
        scenes,
        required,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    return evaluation, summary


def _verify_oof_provenance(
    oof_rows: Sequence[Mapping[str, object]],
    frozen_probe: ProbeArrays,
) -> dict[str, object]:
    """Bind every OOF identity, label, candidate, and Arm-A score to the seal."""

    frozen_probe.validate()
    expected_rows = len(frozen_probe.event_id)
    if len(oof_rows) != expected_rows:
        raise MF3ZUResultAuditError(
            f"OOF/provenance row count differs: {len(oof_rows)} != {expected_rows}"
        )
    for index, row in enumerate(oof_rows):
        count = int(frozen_probe.candidate_mask[index].sum())
        expected_identity = {
            "event_id": str(frozen_probe.event_id[index]),
            "scene_id": str(frozen_probe.scene_id[index]),
            "episode_id": str(frozen_probe.episode_id[index]),
            "decision_step": int(frozen_probe.decision_step[index]),
            "scene_fold": int(frozen_probe.scene_fold[index]),
            "memory_required": bool(frozen_probe.memory_required[index]),
            "candidate_action_ids": list(frozen_probe.candidate_action_ids[index]),
            "target_index": int(frozen_probe.target_index[index]),
        }
        observed_identity = {key: row.get(key) for key in expected_identity}
        if observed_identity != expected_identity:
            changed = [
                key
                for key in expected_identity
                if observed_identity[key] != expected_identity[key]
            ]
            raise MF3ZUResultAuditError(
                f"OOF frozen provenance drift at row {index}: {','.join(changed)}"
            )
        scores = row.get("scores")
        if not isinstance(scores, Mapping):
            raise MF3ZUResultAuditError(f"OOF score mapping missing at row {index}")
        observed_current = scores.get(ARM_CURRENT)
        expected_current = [
            float(value) for value in frozen_probe.base_scores[index, :count]
        ]
        if observed_current != expected_current:
            raise MF3ZUResultAuditError(
                f"OOF frozen ETP score drift at row {index}"
            )
    return {
        "rows": expected_rows,
        "event_order_exact": True,
        "causal_identity_exact": True,
        "candidate_order_from_frozen_evidence_exact": True,
        "target_from_separate_exact_artifact_exact": True,
        "memory_required_from_frozen_evidence_exact": True,
        "ETP_CURRENT_scores_exact": True,
        "evidence_frozen_before_exact_target_open": True,
    }


def audit_result(
    result: Mapping[str, object],
    *,
    oof_rows: Sequence[Mapping[str, object]] | None = None,
    frozen_probe: ProbeArrays | None = None,
    enforce_frozen_counts: bool = True,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    require_frozen_provenance: bool | None = None,
) -> dict[str, object]:
    failures: list[str] = []
    if require_frozen_provenance is None:
        require_frozen_provenance = enforce_frozen_counts
    if result.get("revision") != REVISION:
        failures.append("revision_drift")
    scope = result.get("scope")
    if not isinstance(scope, Mapping) or scope.get("dataset") != "RxR":
        failures.append("not_RxR_only")
    elif scope.get("R2R_evaluated") is not False:
        failures.append("R2R_was_evaluated")
    if result.get("public_split_access") != PUBLIC_CLOSED:
        failures.append("public_split_access")
    for field in (
        "full_navigation_run", "checkpoint_generated",
        "checkpoint_for_deployment", "deployment_authorized",
        "MF3ZT_two_domain_pass_claimed",
    ):
        if result.get(field) is not False:
            failures.append(field)
    if result.get("arms") != list(ARMS):
        failures.append("arm_set_drift")
    metrics = result.get("metrics_per_domain")
    if not isinstance(metrics, Mapping) or set(metrics) != {"RxR"}:
        failures.append("metrics_domain_drift")
    elif not isinstance(metrics["RxR"], Mapping) or set(metrics["RxR"]) != set(ARMS):
        failures.append("metrics_arm_drift")
    else:
        for arm in ARMS:
            if not isinstance(metrics["RxR"][arm], Mapping) or set(metrics["RxR"][arm]) != set(SUBGROUPS):
                failures.append(f"metrics_subgroup_drift:{arm}")

    training = result.get("training")
    if not isinstance(training, Mapping):
        failures.append("training_evidence_missing")
    else:
        if training.get("complete_five_fold_oof") is not True:
            failures.append("OOF_incomplete")
        if training.get("checkpoint_written") is not False:
            failures.append("training_checkpoint_written")
        if training.get("public_split_access") != PUBLIC_CLOSED:
            failures.append("training_public_split_access")
        if training.get("full_navigation_run") is not False:
            failures.append("training_full_navigation")
        for field in (
            "ETP_frozen", "candidate_generator_frozen",
            "visual_backbone_frozen", "topology_encoder_frozen",
        ):
            if training.get(field) is not True:
                failures.append(field)
        schedule = training.get("fixed_schedule")
        expected_schedule = {
            "epochs": 40,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "seed": 20_260_901,
            "early_stopping": False,
            "best_checkpoint_selection": False,
            "model_or_threshold_selection": False,
        }
        if schedule != expected_schedule:
            failures.append("fixed_schedule_drift")
        folds = training.get("folds")
        if not isinstance(folds, list) or len(folds) != FOLDS:
            failures.append("fold_fit_count")
        else:
            held = {row.get("held_fold") for row in folds if isinstance(row, Mapping)}
            if held != set(range(FOLDS)):
                failures.append("held_fold_coverage")
            for row in folds:
                if not isinstance(row, Mapping):
                    failures.append("malformed_fold_fit")
                    continue
                if row.get("normalization_fit_train_fold_only") is not True:
                    failures.append("held_normalization_leakage")
                if row.get("B_C_common_initialization") is not True:
                    failures.append("B_C_initialization_drift")
                if row.get("B_C_common_batch_order") is not True:
                    failures.append("B_C_batch_order_drift")
                shuffled = row.get("shuffled_memory")
                if not isinstance(shuffled, Mapping):
                    failures.append("shuffled_memory_evidence_missing")
                elif not (
                    shuffled.get("train_derangement") is True
                    and shuffled.get("held_donors_train_only") is True
                    and shuffled.get("outcome_or_target_used") is False
                ):
                    failures.append("shuffled_memory_leakage")

    provenance_evidence: dict[str, object] | None = None
    if require_frozen_provenance:
        if oof_rows is None or frozen_probe is None:
            failures.append("frozen_OOF_provenance_not_supplied_to_audit")
        else:
            try:
                provenance_evidence = _verify_oof_provenance(
                    oof_rows, frozen_probe
                )
            except (IndexError, KeyError, TypeError, ValueError, MF3ZUTrainingError, MF3ZUResultAuditError) as error:
                failures.append(f"frozen_OOF_provenance_failed:{error}")

    recomputed_evaluation: dict[str, object] | None = None
    recomputed_summary: dict[str, int] | None = None
    if oof_rows is None:
        failures.append("OOF_predictions_not_supplied_to_audit")
    else:
        try:
            recomputed_evaluation, recomputed_summary = _recompute_oof_evaluation(
                oof_rows,
                enforce_frozen_counts=enforce_frozen_counts,
                bootstrap_replicates=bootstrap_replicates,
            )
        except (KeyError, TypeError, ValueError, MF3ZUResultAuditError) as error:
            failures.append(f"OOF_recomputation_failed:{error}")

    expected_gates = None
    if recomputed_evaluation is not None:
        expected_metrics = recomputed_evaluation["metrics"]
        if not isinstance(metrics, Mapping) or metrics.get("RxR") != expected_metrics:
            failures.append("metrics_do_not_match_OOF")
        if result.get("pairwise_deltas") != recomputed_evaluation["pairwise_deltas"]:
            failures.append("pairwise_deltas_do_not_match_OOF")
        if result.get("scene_bootstrap_CI") != recomputed_evaluation["scene_bootstrap_CI"]:
            failures.append("scene_bootstrap_does_not_match_OOF")
        expected_support = recomputed_evaluation["subgroup_support"]["MEMORY_REQUIRED"]
        if result.get("memory_required") != expected_support:
            failures.append("memory_required_support_does_not_match_OOF")
        expected_gates = apply_fixed_rxr_gates(recomputed_evaluation)
        if result.get("fixed_gates") != expected_gates:
            failures.append("fixed_gate_recomputation_mismatch")
        if result.get("status") != expected_gates["status"]:
            failures.append("status_mismatch")
        if result.get("final_PASS_FAIL") != expected_gates["final_PASS_FAIL"]:
            failures.append("final_PASS_FAIL_mismatch")

    population = result.get("population")
    if not isinstance(population, Mapping):
        failures.append("population_summary_missing")
    elif recomputed_summary is not None:
        if int(population.get("rankable_exact_target_decisions", -1)) != recomputed_summary["rows"]:
            failures.append("population_rankable_count_mismatch")
        if int(population.get("episodes", -1)) != recomputed_summary["episodes"]:
            failures.append("population_episode_count_mismatch")
        if int(population.get("raw_scenes", -1)) != recomputed_summary["raw_scenes"]:
            failures.append("population_scene_count_mismatch")
        if enforce_frozen_counts and int(population.get("target_blind_decisions", -1)) != EXPECTED_POPULATION_ROWS:
            failures.append("population_target_blind_count_drift")

    return {
        "schema_version": "revealnav-mf3zu-rxr-feasibility-result-audit/1",
        "revision": REVISION,
        "status": "MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT_PASS" if not failures else "MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT_FAIL",
        "passed": not failures,
        "failures": failures,
        "fixed_gates_recomputed": expected_gates,
        "OOF_recomputed_support": recomputed_summary,
        "OOF_metrics_recomputed": recomputed_evaluation is not None,
        "frozen_OOF_provenance": provenance_evidence,
        "frozen_OOF_provenance_verified": provenance_evidence is not None,
        "formal_frozen_counts_enforced": bool(enforce_frozen_counts),
        "bootstrap_replicates_recomputed": int(bootstrap_replicates),
        "bootstrap_seed_recomputed": BOOTSTRAP_SEED,
        "public_split_access": dict(PUBLIC_CLOSED),
        "full_navigation_run": False,
        "checkpoint_generated": False,
    }


def _write_once(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise MF3ZUResultAuditError(f"immutable audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUResultAuditError(f"stale audit partial exists: {partial}")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    try:
        verify_protocol()
        if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink():
            raise MF3ZUResultAuditError("immutable MF3ZU result is missing")
        result = _object(RESULT_PATH)
        inventory = result.get("OOF_predictions")
        if not isinstance(inventory, Mapping):
            raise MF3ZUResultAuditError("OOF inventory is missing")
        oof_rows = inventory.get("rows")
        if (
            inventory.get("storage") != "embedded_in_result"
            or inventory.get("complete") is not True
            or not isinstance(oof_rows, list)
            or any(not isinstance(row, Mapping) for row in oof_rows)
            or int(inventory.get("row_count", -1)) != len(oof_rows)
        ):
            raise MF3ZUResultAuditError("embedded OOF inventory drift")
        canonical_oof = _canonical_oof_jsonl(oof_rows)
        if (
            int(inventory.get("canonical_jsonl_bytes", -1)) != len(canonical_oof)
            or inventory.get("canonical_jsonl_sha256")
            != hashlib.sha256(canonical_oof).hexdigest()
        ):
            raise MF3ZUResultAuditError("embedded OOF canonical inventory drift")
        # The shared loader proves the evidence manifest is frozen and
        # target-blind before it opens the separate exact-target artifact.
        frozen_probe = load_frozen_probe_inputs()
        audit = audit_result(
            result,
            oof_rows=oof_rows,
            frozen_probe=frozen_probe,
            enforce_frozen_counts=True,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            require_frozen_provenance=True,
        )
        audit["result"] = {
            "path": str(RESULT_PATH.relative_to(ROOT)),
            "bytes": RESULT_PATH.stat().st_size,
            "sha256": sha256_file(RESULT_PATH),
        }
        _write_once(AUDIT_PATH, audit)
    except (
        OSError, KeyError, TypeError, ValueError, ProtocolError,
        MF3ZUTrainingError, MF3ZUResultAuditError,
    ) as error:
        print(f"MF3ZU_RXR_RESULT_AUDIT_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
