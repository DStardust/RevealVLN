"""Independent visual-review labels and the fixed exploratory REE probe.

This revision replaces the inadmissible cached-Qwen proxy with labels entered
after direct inspection of every causal panorama.  The labels are AI review,
not human review or gold.  No function in this module reads Qwen factor, UAD,
or rationale records.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .codex_proxy_ree import (
    EPOCHS,
    FOLDS,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
    ProxyExample,
    _arm_metrics,
    _fit_predict_fold,
    causal_feature_rows,
    scene_fold,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zp_codex_visual_review_v1"
OUTPUT = ROOT / "artifacts/training/mf3zp_codex_visual_review_v1"

PRIMARY_PROTOCOL = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_PROTOCOL.json"
SOURCE = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_SOURCE.json"
LABELS = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_LABELS.jsonl"
LABEL_AUDIT = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_LABEL_AUDIT.json"
LABEL_MANIFEST = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_LABEL_MANIFEST.json"
TRAINING_PROTOCOL = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_TRAINING_PROTOCOL.json"
RESULT = OUTPUT / "MF3ZP_CODEX_VISUAL_REVIEW_REE_RESULT.json"

SELECTION = ROOT / "artifacts/training/mf3zp_single_expert_dec_scout_v1/MF3ZP_SINGLE_EXPERT_DEC_SCOUT_SELECTION.json"
REVIEW_TEMPLATE = ROOT / "artifacts/training/mf3zp_single_expert_dec_scout_v1/MF3ZP_SINGLE_EXPERT_REVIEW_TEMPLATE.jsonl"

PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}
ROLES = {
    "DEC_REQUIRED",
    "PREREQUISITE_ONLY",
    "FUTURE_NOT_RELEVANT",
    "REDUNDANT",
    "INCORRECT",
}
TRAINED_ROLES = {"DEC_REQUIRED", "PREREQUISITE_ONLY"}
FACTOR_ENCODINGS = {"000", "100", "101", "110", "111"}
FORBIDDEN_PATH_PARTS = (
    "/qwen_preannotations/evidence/",
    "/qwen_preannotations/evidence_v1_1/",
)
FORBIDDEN_OUTCOME_KEYS = {
    "delta_utility",
    "reward",
    "outcome",
    "success",
    "spl",
    "ndtw",
    "sdtw",
    "catastrophe",
    "correct_action",
    "native_role",
    "runner_role",
}


class VisualReviewError(RuntimeError):
    """Fail-closed error for the independent-review path."""


def stable_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(stable_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or resolved == root
        or root not in resolved.parents
    ):
        raise VisualReviewError(f"invalid project-local file: {path}")
    relative = resolved.relative_to(root).as_posix()
    if any(part in f"/{relative}/" for part in FORBIDDEN_PATH_PARTS):
        raise VisualReviewError(f"forbidden Qwen factor cache: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisualReviewError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise VisualReviewError(f"nonempty JSONL objects required: {path}")
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise VisualReviewError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise VisualReviewError(f"stale partial output: {partial}")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def _reject_outcomes(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in FORBIDDEN_OUTCOME_KEYS or key.startswith(
                ("reward_", "outcome_", "treatment_")
            ):
                raise VisualReviewError(f"outcome field at {path}.{raw_key}")
            _reject_outcomes(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcomes(child, f"{path}[{index}]")
    elif isinstance(value, str):
        normalized = "/" + value.replace("\\", "/").strip("/") + "/"
        if any(part in normalized for part in FORBIDDEN_PATH_PARTS):
            raise VisualReviewError(f"forbidden Qwen factor cache at {path}")


def _decode_factor(value: object) -> tuple[bool, bool, bool]:
    text = str(value)
    if text not in FACTOR_ENCODINGS:
        raise VisualReviewError(f"invalid factor encoding: {value!r}")
    return tuple(character == "1" for character in text)  # type: ignore[return-value]


def _verify_primary_protocol() -> dict[str, object]:
    protocol = read_json(PRIMARY_PROTOCOL)
    if protocol.get("status") != "SEALED_BEFORE_CODEX_VISUAL_LABELS_AND_TRAINING":
        raise VisualReviewError("primary protocol status drift")
    if protocol.get("public_split_access") != PUBLIC_CLOSED:
        raise VisualReviewError("primary protocol opened a public split")
    if any(protocol.get("authorization", {}).values()):
        raise VisualReviewError("primary protocol opened downstream authorization")
    for key in ("selection", "review_template"):
        expected = protocol["inputs"][key]
        path = ROOT / str(expected["path"])
        if inventory(path) != expected:
            raise VisualReviewError(f"primary protocol input drift: {key}")
    return protocol


def causal_image_inventory(
    templates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paths: set[Path] = set()
    for row in templates:
        for prefix in row["prefixes"]:
            paths.add(ROOT / str(prefix["causal_storyboard_path"]))
            paths.add(ROOT / str(prefix["current_panorama_path"]))
    values = [inventory(path) for path in sorted(paths)]
    return {
        "count": len(values),
        "total_bytes": sum(int(value["bytes"]) for value in values),
        "sha256": stable_sha256(values),
    }


def validate_manual_event(
    manual: Mapping[str, object], template: Mapping[str, object]
) -> dict[str, object]:
    if str(manual.get("event_id")) != str(template["event_id"]):
        raise VisualReviewError("manual/template event identity mismatch")
    graph = template["constraint_graph"]
    graph_ids = [str(row["constraint_id"]) for row in graph]
    if len(graph_ids) != len(set(graph_ids)):
        raise VisualReviewError("constraint graph contains duplicate IDs")
    explicit = manual.get("roles", {})
    if not isinstance(explicit, Mapping) or not set(map(str, explicit)) <= set(graph_ids):
        raise VisualReviewError("manual roles do not match frozen graph")
    default = manual.get("default_role")
    roles = {
        cid: str(explicit.get(cid, default))
        for cid in graph_ids
    }
    if any(role not in ROLES for role in roles.values()):
        raise VisualReviewError("every frozen constraint needs one valid role")

    steps = [int(prefix["step"]) for prefix in template["prefixes"]]
    if not steps or steps != sorted(steps) or steps[-1] > int(template["decision_step"]):
        raise VisualReviewError("review prefixes are not strictly causal")
    expected_steps = {str(step) for step in steps}
    raw_factors = manual.get("factors", {})
    if not isinstance(raw_factors, Mapping):
        raise VisualReviewError("manual factors must be an object")
    constraints: dict[str, object] = {}
    for cid, role in roles.items():
        values = raw_factors.get(cid)
        if role in TRAINED_ROLES:
            if not isinstance(values, Mapping) or set(map(str, values)) != expected_steps:
                raise VisualReviewError(f"incomplete factor sequence: {cid}")
            factors = []
            for step in steps:
                instantiated, distinguishable, resolved = _decode_factor(
                    values[str(step)]
                )
                factors.append({
                    "step": step,
                    "instantiated": instantiated,
                    "distinguishable": distinguishable,
                    "resolved": resolved,
                })
        else:
            if values is not None:
                raise VisualReviewError(f"non-DEC constraint has factors: {cid}")
            factors = [
                {
                    "step": step,
                    "instantiated": None,
                    "distinguishable": None,
                    "resolved": None,
                }
                for step in steps
            ]
        constraints[cid] = {"dec_role": role, "factor_by_step": factors}

    missing = manual.get("missing_dec_constraints", [])
    if not isinstance(missing, list):
        raise VisualReviewError("missing DEC constraints must be a list")
    extra_graph: list[dict[str, object]] = []
    for item in missing:
        if not isinstance(item, Mapping):
            raise VisualReviewError("malformed missing DEC constraint")
        cid = str(item.get("human_dec_item_id", ""))
        role = str(item.get("role", ""))
        values = item.get("factors")
        if (
            not cid.startswith("missing::")
            or cid in constraints
            or role not in TRAINED_ROLES
            or not isinstance(values, Mapping)
            or set(map(str, values)) != expected_steps
        ):
            raise VisualReviewError("invalid missing DEC constraint")
        factors = []
        for step in steps:
            instantiated, distinguishable, resolved = _decode_factor(
                values[str(step)]
            )
            factors.append({
                "step": step,
                "instantiated": instantiated,
                "distinguishable": distinguishable,
                "resolved": resolved,
            })
        constraints[cid] = {"dec_role": role, "factor_by_step": factors}
        extra_graph.append({
            "constraint_id": cid,
            "kind": str(item.get("kind", "ENTITY")),
            "subject": "independent_visual_review",
            "relation": None,
            "object": str(item.get("text", "")),
            "dependencies": [],
            "decisive_for": [],
        })
    return {
        "roles": roles,
        "constraints": constraints,
        "extra_graph": extra_graph,
        "steps": steps,
    }


def _prefix_sources(template: Mapping[str, object]) -> list[dict[str, object]]:
    result = []
    for prefix in template["prefixes"]:
        step = int(prefix["step"])
        panorama = ROOT / str(prefix["current_panorama_path"])
        storyboard = ROOT / str(prefix["causal_storyboard_path"])
        run_dir = panorama.parents[1]
        trace_path = run_dir / "causal_prefix_records.jsonl"
        trace = {
            int(row["step"]): row
            for row in read_jsonl(trace_path)
        }
        if step not in trace:
            raise VisualReviewError("causal array reference is missing")
        arrays = ROOT / str(trace[step]["arrays"]["path"])
        if sha256_file(arrays) != str(trace[step]["arrays"]["sha256"]):
            raise VisualReviewError("causal array hash drift")
        result.append({
            "step": step,
            "candidate_ids": list(prefix["candidate_ids"]),
            "current_panorama": inventory(panorama),
            "causal_storyboard": inventory(storyboard),
            "causal_trace": inventory(trace_path),
            "arrays": inventory(arrays),
        })
    return result


def materialize_visual_labels() -> dict[str, object]:
    protocol = _verify_primary_protocol()
    source = read_json(SOURCE)
    templates = read_jsonl(REVIEW_TEMPLATE)
    selection = read_json(SELECTION)
    _reject_outcomes(source)
    if (
        source.get("revision") != REVISION
        or source.get("provenance")
        != "CODEX_INDEPENDENT_VISUAL_REVIEW_NOT_HUMAN_NOT_GOLD"
        or source.get("labels_complete") is not True
        or source.get("expected_events") != 80
    ):
        raise VisualReviewError("independent visual source is incomplete")
    review_policy = source.get("review_policy", {})
    if any(
        review_policy.get(key) is not False
        for key in (
            "qwen_factor_labels_read",
            "qwen_uad_labels_read",
            "qwen_rationales_read",
            "old_training_results_used",
        )
    ):
        raise VisualReviewError("independent-review isolation flag drift")
    if causal_image_inventory(templates) != protocol["inputs"]["causal_image_inventory"]:
        raise VisualReviewError("causal image inventory drift")
    selected = selection.get("events")
    if not isinstance(selected, list) or len(selected) != 80:
        raise VisualReviewError("frozen selection drift")
    selected_ids = [str(row["event_id"]) for row in selected]
    if selected_ids != [str(row["event_id"]) for row in templates]:
        raise VisualReviewError("selection/template order drift")
    manual_rows = source.get("events")
    if not isinstance(manual_rows, list) or len(manual_rows) != 80:
        raise VisualReviewError("manual event count drift")
    if [int(row["index"]) for row in manual_rows] != list(range(80)):
        raise VisualReviewError("manual event indices drift")
    if [str(row["event_id"]) for row in manual_rows] != selected_ids:
        raise VisualReviewError("manual event order drift")
    if Counter(str(row["dataset"]) for row in templates) != {"R2R": 40, "RxR": 40}:
        raise VisualReviewError("visual review population must remain 40/40")

    rows: list[dict[str, object]] = []
    roles = Counter()
    factor_rows = 0
    missing_count = 0
    array_paths: set[Path] = set()
    for manual, template in zip(manual_rows, templates, strict=True):
        validated = validate_manual_event(manual, template)
        prefixes = _prefix_sources(template)
        for prefix in prefixes:
            array_paths.add(ROOT / str(prefix["arrays"]["path"]))
        constraints = validated["constraints"]
        roles.update(item["dec_role"] for item in constraints.values())
        factor_rows += sum(
            len(item["factor_by_step"])
            for item in constraints.values()
            if item["dec_role"] in TRAINED_ROLES
        )
        missing_count += len(validated["extra_graph"])
        row = {
            "schema_version": "revealnav-mf3zp-codex-independent-visual-label/1",
            "revision": REVISION,
            "provenance": {
                "label_class": "CODEX_INDEPENDENT_VISUAL_REVIEW_NOT_HUMAN_NOT_GOLD",
                "factor_source": "direct_causal_panorama_inspection_by_codex",
                "dec_source": "direct_instruction_graph_and_causal_panorama_review_by_codex",
                "human_verified": False,
                "gold": False,
                "qwen_factor_cache_files_read": 0,
                "qwen_api_calls": 0,
            },
            "event_id": str(template["event_id"]),
            "dataset": str(template["dataset"]),
            "scene_id": str(template["scene_id"]),
            "episode_id": str(template["episode_id"]),
            "instruction": str(template["instruction"]),
            "decision_step": int(template["decision_step"]),
            "constraint_graph_sha256": str(template["constraint_graph_sha256"]),
            "constraint_graph": template["constraint_graph"],
            "independent_missing_constraints": validated["extra_graph"],
            "constraints": constraints,
            "prefix_sources": prefixes,
            "review_note": str(manual.get("review_note", "")),
            "public_split_access": PUBLIC_CLOSED,
        }
        _reject_outcomes(row)
        rows.append(row)

    atomic_jsonl(LABELS, rows)
    audit = {
        "schema_version": "revealnav-mf3zp-codex-visual-label-audit/1",
        "revision": REVISION,
        "status": "CODEX_INDEPENDENT_VISUAL_LABEL_AUDIT_PASS",
        "events": len(rows),
        "domains": dict(Counter(str(row["dataset"]) for row in rows)),
        "scenes": len({str(row["scene_id"]) for row in rows}),
        "prefixes": sum(len(row["prefix_sources"]) for row in rows),
        "trained_constraint_sequences": sum(roles[role] for role in TRAINED_ROLES),
        "factor_rows": factor_rows,
        "role_counts": dict(sorted(roles.items())),
        "independent_missing_constraints": missing_count,
        "qwen_factor_cache_files_read": 0,
        "qwen_api_calls": 0,
        "human_labels_fabricated": False,
        "formal_label_validity_pass": False,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    atomic_json(LABEL_AUDIT, audit)
    arrays = [inventory(path) for path in sorted(array_paths)]
    manifest = {
        "schema_version": "revealnav-mf3zp-codex-visual-label-manifest/1",
        "revision": REVISION,
        "status": "CODEX_INDEPENDENT_VISUAL_LABELS_COMPLETE_NOT_HUMAN_NOT_GOLD",
        "source": inventory(SOURCE),
        "labels": inventory(LABELS),
        "audit": inventory(LABEL_AUDIT),
        "causal_array_inventory": {
            "count": len(arrays),
            "sha256": stable_sha256(arrays),
        },
        "qwen_factor_cache_files_read": 0,
        "qwen_api_calls": 0,
        "human_labels_fabricated": False,
        "formal_label_validity_pass": False,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    atomic_json(LABEL_MANIFEST, manifest)
    return manifest


def seal_training_protocol() -> dict[str, object]:
    _verify_primary_protocol()
    audit = read_json(LABEL_AUDIT)
    manifest = read_json(LABEL_MANIFEST)
    if audit.get("status") != "CODEX_INDEPENDENT_VISUAL_LABEL_AUDIT_PASS":
        raise VisualReviewError("training requires a complete independent-label audit")
    if any(audit.get("public_split_access", {}).values()):
        raise VisualReviewError("label audit opened a public split")
    protocol = {
        "schema_version": "revealnav-mf3zp-codex-visual-ree-training-protocol/1",
        "revision": REVISION,
        "status": "SEALED_AFTER_LABEL_COMPLETION_BEFORE_TRAINING_RESULT",
        "purpose": "exploratory independent-visual-label temporal learnability",
        "inputs": {
            "primary_protocol": inventory(PRIMARY_PROTOCOL),
            "manual_source": inventory(SOURCE),
            "labels": inventory(LABELS),
            "label_audit": inventory(LABEL_AUDIT),
            "label_manifest": inventory(LABEL_MANIFEST),
        },
        "training": {
            "folds": FOLDS,
            "fold_function": "historical_fixed_mf3zp_codex_proxy_ree_fold_v1",
            "raw_scene_disjoint": True,
            "arms": ["snapshot", "temporal"],
            "architecture": "TemporalRevealExpiryEncoder_hidden_64",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "prediction_boundary": 0.5,
            "stability_k": 3,
            "positive_signal_rule": (
                "each_domain_factor_nll_improvement_gt_0_and_"
                "each_domain_uad_macro_f1_improvement_gt_0"
            ),
            "hyperparameter_search": False,
            "checkpoint_generation": False,
        },
        "authorization": {
            "formal_label_validity": False,
            "oracle_headroom": False,
            "formal_ree": False,
            "skill_policy": False,
            "deployment_checkpoint": False,
        },
        "public_split_access": PUBLIC_CLOSED,
    }
    if manifest.get("labels") != protocol["inputs"]["labels"]:
        raise VisualReviewError("manifest/label inventory mismatch")
    atomic_json(TRAINING_PROTOCOL, protocol)
    return protocol


def verify_training_protocol() -> dict[str, object]:
    protocol = read_json(TRAINING_PROTOCOL)
    if protocol.get("status") != "SEALED_AFTER_LABEL_COMPLETION_BEFORE_TRAINING_RESULT":
        raise VisualReviewError("training protocol status drift")
    if protocol.get("public_split_access") != PUBLIC_CLOSED:
        raise VisualReviewError("training protocol opened a public split")
    if any(protocol.get("authorization", {}).values()):
        raise VisualReviewError("training protocol opened downstream authorization")
    for name, expected in protocol["inputs"].items():
        if inventory(ROOT / str(expected["path"])) != expected:
            raise VisualReviewError(f"training input drift: {name}")
    expected_training = {
        "folds": FOLDS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "prediction_boundary": 0.5,
        "stability_k": 3,
        "hyperparameter_search": False,
        "checkpoint_generation": False,
    }
    training = protocol.get("training", {})
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise VisualReviewError("fixed training configuration drift")
    return protocol


def load_visual_examples() -> list[ProxyExample]:
    verify_training_protocol()
    examples: list[ProxyExample] = []
    for row in read_jsonl(LABELS):
        _reject_outcomes(row)
        graph = {
            str(item["constraint_id"]): item
            for item in (
                list(row["constraint_graph"])
                + list(row["independent_missing_constraints"])
            )
        }
        for cid, item in row["constraints"].items():
            if item["dec_role"] not in TRAINED_ROLES:
                continue
            targets = np.asarray([
                [
                    factor["instantiated"],
                    factor["distinguishable"],
                    factor["resolved"],
                ]
                for factor in item["factor_by_step"]
            ], dtype=np.float32)
            features = causal_feature_rows(row["prefix_sources"], graph[str(cid)])
            if targets.shape != (len(features), 3):
                raise VisualReviewError("visual feature/target alignment drift")
            examples.append(ProxyExample(
                event_id=str(row["event_id"]),
                dataset=str(row["dataset"]),
                scene_id=str(row["scene_id"]),
                constraint_id=str(cid),
                features=features,
                targets=targets,
            ))
    if not examples:
        raise VisualReviewError("independent label set has no trainable constraints")
    return examples


def train_visual_ree(device_name: str) -> dict[str, object]:
    protocol = verify_training_protocol()
    if RESULT.exists() or RESULT.is_symlink():
        raise VisualReviewError("refusing to overwrite the one-shot result")
    examples = load_visual_examples()
    scenes = {example.scene_id for example in examples}
    fold_scenes = {
        fold: sorted(scene for scene in scenes if scene_fold(scene) == fold)
        for fold in range(FOLDS)
    }
    if any(not held for held in fold_scenes.values()):
        raise VisualReviewError("scene fold is empty")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise VisualReviewError("CUDA requested but unavailable")
    predictions: dict[str, list[np.ndarray | None]] = {
        "snapshot": [None] * len(examples),
        "temporal": [None] * len(examples),
    }
    fold_reports = []
    for fold in range(FOLDS):
        held = set(fold_scenes[fold])
        train_indices = [
            index for index, example in enumerate(examples)
            if example.scene_id not in held
        ]
        evaluate_indices = [
            index for index, example in enumerate(examples)
            if example.scene_id in held
        ]
        if (
            not train_indices
            or not evaluate_indices
            or {examples[index].scene_id for index in train_indices} & held
        ):
            raise VisualReviewError("scene-disjoint fold construction failed")
        report = {
            "fold": fold,
            "held_scenes": sorted(held),
            "train_examples": len(train_indices),
            "evaluate_examples": len(evaluate_indices),
            "arms": {},
        }
        for arm in ("snapshot", "temporal"):
            values, loss = _fit_predict_fold(
                [examples[index] for index in train_indices],
                [examples[index] for index in evaluate_indices],
                arm=arm,
                fold=fold,
                device=device,
            )
            for index, value in zip(evaluate_indices, values, strict=True):
                predictions[arm][index] = value
            report["arms"][arm] = {"final_training_loss": loss}
        fold_reports.append(report)
    if any(value is None for arm in predictions.values() for value in arm):
        raise VisualReviewError("OOF predictions are incomplete")

    metrics = {}
    domain_positive = []
    for domain in ("R2R", "RxR"):
        snapshot = _arm_metrics(
            examples, predictions["snapshot"], domain  # type: ignore[arg-type]
        )
        temporal = _arm_metrics(
            examples, predictions["temporal"], domain  # type: ignore[arg-type]
        )
        delta_nll = float(snapshot["factor_mean_nll"] - temporal["factor_mean_nll"])
        delta_uad = float(temporal["uad_macro_f1"] - snapshot["uad_macro_f1"])
        positive = delta_nll > 0.0 and delta_uad > 0.0
        domain_positive.append(positive)
        metrics[domain] = {
            "snapshot": snapshot,
            "temporal": temporal,
            "temporal_minus_snapshot": {
                "factor_nll_improvement": delta_nll,
                "uad_macro_f1_improvement": delta_uad,
            },
            "independent_visual_temporal_signal_positive": positive,
        }
    signal = (
        "EXPLORATORY_INDEPENDENT_VISUAL_TEMPORAL_SIGNAL_POSITIVE"
        if all(domain_positive)
        else "EXPLORATORY_INDEPENDENT_VISUAL_TEMPORAL_SIGNAL_MIXED_OR_NEGATIVE"
    )
    result = {
        "schema_version": "revealnav-mf3zp-codex-visual-ree-result/1",
        "revision": REVISION,
        "status": "EXPLORATORY_CODEX_INDEPENDENT_VISUAL_REE_COMPLETE",
        "signal_status": signal,
        "protocol_sha256": sha256_file(TRAINING_PROTOCOL),
        "label_manifest_sha256": sha256_file(LABEL_MANIFEST),
        "events": len({example.event_id for example in examples}),
        "constraint_sequences": len(examples),
        "scenes": len(scenes),
        "domains": dict(Counter(example.dataset for example in examples)),
        "feature_dim": int(examples[0].features.shape[1]),
        "folds": fold_reports,
        "metrics": metrics,
        "limitations": [
            "labels are independent Codex visual review, not human labels or gold",
            "instruction constraint graphs remain frozen provisional audit objects",
            "expiry supervision is unavailable and was not fabricated",
            "this result cannot authorize Oracle Headroom, formal REE, skill policy, or public evaluation",
        ],
        "qwen_factor_cache_files_read": 0,
        "qwen_api_calls": 0,
        "human_labels_fabricated": False,
        "formal_label_validity_pass": False,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
        "device": str(device),
        "source_protocol": protocol["status"],
    }
    atomic_json(RESULT, result)
    return result


__all__ = [
    "VisualReviewError",
    "causal_image_inventory",
    "load_visual_examples",
    "materialize_visual_labels",
    "seal_training_protocol",
    "train_visual_ree",
    "validate_manual_event",
    "verify_training_protocol",
]
