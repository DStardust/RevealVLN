"""Outcome-blind AI-proxy labels and fixed REE learnability probe.

This module deliberately lives outside the human-review implementation.  It
never writes a human reviewer identity, gold status, downstream authorization,
or a deployment checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .evidence_uad import derive_constraint_uad
from .temporal_uad_model import TemporalRevealExpiryEncoder


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zp_codex_proxy_ree_v1"
OUTPUT = ROOT / "artifacts/training/mf3zp_codex_proxy_ree_v1"

SELECTION = ROOT / "artifacts/training/mf3zp_single_expert_dec_scout_v1/MF3ZP_SINGLE_EXPERT_DEC_SCOUT_SELECTION.json"
REVIEW_TEMPLATE = ROOT / "artifacts/training/mf3zp_single_expert_dec_scout_v1/MF3ZP_SINGLE_EXPERT_REVIEW_TEMPLATE.jsonl"
PILOT_EVENTS = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEAL_EVENTS.jsonl"
SOURCE_REQUESTS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_ANNOTATION_REQUESTS.jsonl"
EVIDENCE_ROOT = ROOT / "artifacts/training/mf3zp_revealskill_v1/qwen_preannotations"

LABELS = OUTPUT / "MF3ZP_CODEX_PROXY_LABELS.jsonl"
LABEL_MANIFEST = OUTPUT / "MF3ZP_CODEX_PROXY_LABEL_MANIFEST.json"
PROTOCOL = OUTPUT / "MF3ZP_CODEX_PROXY_REE_PROTOCOL.json"
RESULT = OUTPUT / "MF3ZP_CODEX_PROXY_REE_RESULT.json"

PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}
FORBIDDEN_LABEL_KEYS = {
    "delta_utility", "reward", "outcome", "success", "spl", "ndtw",
    "sdtw", "catastrophe", "correct_action", "native_role", "runner_role",
}
DEC_ROLES = {"DEC_REQUIRED", "PREREQUISITE_ONLY"}

FOLDS = 5
EPOCHS = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 20260901
PROJECTION_DIM = 16
CONSTRAINT_DIM = 32


class ProxyREEError(RuntimeError):
    pass


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
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise ProxyREEError(f"invalid project-local input: {path}")
    return {
        "path": str(resolved.relative_to(ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProxyREEError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ProxyREEError(f"nonempty JSONL objects required: {path}")
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProxyREEError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProxyREEError(f"stale partial output: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def _reject_outcomes(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in FORBIDDEN_LABEL_KEYS or key.startswith(("reward_", "outcome_", "treatment_")):
                raise ProxyREEError(f"outcome-bearing label field at {path}.{raw_key}")
            _reject_outcomes(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcomes(child, f"{path}[{index}]")


def _evidence_index() -> dict[str, tuple[Path, dict[str, object]]]:
    result: dict[str, tuple[Path, dict[str, object]]] = {}
    for directory in ("evidence", "evidence_v1_1"):
        for path in sorted((EVIDENCE_ROOT / directory).glob("*.json")):
            record = read_json(path)
            request_id = str(record.get("source_request_id", ""))
            if not request_id:
                raise ProxyREEError(f"evidence source request missing: {path}")
            if record.get("human_verified") is not False or record.get("gold") is not False:
                raise ProxyREEError("proxy source must remain provisional machine annotation")
            result[request_id] = (path, record)
    return result


def _graph_maps(graph: Sequence[Mapping[str, object]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    ids = {str(row["constraint_id"]) for row in graph}
    if len(ids) != len(graph):
        raise ProxyREEError("constraint IDs must be unique")
    dependencies = {
        str(row["constraint_id"]): {str(value) for value in row.get("dependencies", [])}
        for row in graph
    }
    if any(not values <= ids for values in dependencies.values()):
        raise ProxyREEError("constraint dependency is missing")
    descendants = {cid: set() for cid in ids}
    for child, parents in dependencies.items():
        for parent in parents:
            descendants[parent].add(child)
    changed = True
    while changed:
        changed = False
        for cid in ids:
            expanded = set().union(*(descendants[value] for value in tuple(descendants[cid]))) if descendants[cid] else set()
            if not expanded <= descendants[cid]:
                descendants[cid].update(expanded)
                changed = True
    return dependencies, descendants


def infer_proxy_dec_roles(
    graph: Sequence[Mapping[str, object]],
    factors_by_step: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, str]:
    """Infer one deterministic proxy DEC without claiming human judgment."""

    if not factors_by_step:
        raise ProxyREEError("proxy DEC requires at least one causal prefix")
    dependencies, descendants = _graph_maps(graph)
    ids = set(dependencies)
    steps = sorted(factors_by_step)
    if any(set(factors_by_step[step]) != ids for step in steps):
        raise ProxyREEError("factor/graph population mismatch")

    def signaled(value: Mapping[str, object]) -> bool:
        return bool(
            value.get("instantiated")
            or value.get("distinguishable")
            or value.get("resolved")
            or value.get("candidate_ids")
            or value.get("evidence_image_indices")
        )

    current = {cid for cid, value in factors_by_step[steps[-1]].items() if signaled(value)}
    if not current:
        current = {
            cid for cid in ids
            if any(signaled(factors_by_step[step][cid]) for step in steps)
        }
    if not current:
        raise ProxyREEError("Qwen evidence contains no outcome-blind decision signal")
    terminals = {
        cid for cid in current if not (descendants[cid] & current)
    }
    closure = set(terminals)
    frontier = list(terminals)
    while frontier:
        child = frontier.pop()
        for parent in dependencies[child]:
            if parent not in closure:
                closure.add(parent)
                frontier.append(parent)

    roles: dict[str, str] = {}
    decision = steps[-1]
    for cid in ids:
        if cid not in closure:
            roles[cid] = "FUTURE_NOT_RELEVANT"
            continue
        if cid in terminals:
            roles[cid] = "DEC_REQUIRED"
            continue
        latest_signal = max(
            (step for step in steps if signaled(factors_by_step[step][cid])),
            default=decision,
        )
        roles[cid] = (
            "PREREQUISITE_ONLY" if latest_signal < decision else "DEC_REQUIRED"
        )
    if not any(value == "DEC_REQUIRED" for value in roles.values()):
        raise ProxyREEError("proxy DEC has no required constraint")
    return roles


def build_proxy_labels() -> dict[str, object]:
    selection = read_json(SELECTION)
    selected = selection.get("events")
    if not isinstance(selected, list) or len(selected) != 80:
        raise ProxyREEError("frozen 80-event selection drift")
    selected_ids = [str(row["event_id"]) for row in selected]
    if Counter(str(row["dataset"]) for row in selected) != {"R2R": 40, "RxR": 40}:
        raise ProxyREEError("proxy population must remain 40/40")

    templates = {str(row["event_id"]): row for row in read_jsonl(REVIEW_TEMPLATE)}
    events = {str(row["event_id"]): row for row in read_jsonl(PILOT_EVENTS)}
    requests = {
        (
            str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]),
            str(row["event_id"]), int(row["prefix_step"]),
        ): row
        for row in read_jsonl(SOURCE_REQUESTS)
    }
    evidence = _evidence_index()
    rows: list[dict[str, object]] = []
    source_evidence_paths: set[Path] = set()
    arrays_paths: set[Path] = set()

    for event_id in selected_ids:
        template = templates[event_id]
        event = events[event_id]
        graph = template["constraint_graph"]
        graph_ids = {str(row["constraint_id"]) for row in graph}
        factors_by_step: dict[int, dict[str, dict[str, object]]] = {}
        prefix_sources = []
        for prefix in template["prefixes"]:
            step = int(prefix["step"])
            key = (
                str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]),
                str(event["source_observation_stream_id"]), step,
            )
            request = requests.get(key)
            if request is None:
                raise ProxyREEError(f"causal prefix request missing: {key}")
            request_id = str(request["request_id"])
            if request_id not in evidence:
                raise ProxyREEError(f"cached evidence missing: {request_id}")
            evidence_path, evidence_record = evidence[request_id]
            source_evidence_paths.add(evidence_path)
            normalized = evidence_record.get("normalized_constraints")
            if not isinstance(normalized, Mapping) or set(map(str, normalized)) != graph_ids:
                raise ProxyREEError("cached evidence graph mismatch")
            factors_by_step[step] = {
                str(cid): {
                    "instantiated": bool(value["instantiated"]),
                    "distinguishable": bool(value["distinguishable"]),
                    "resolved": bool(value["resolved"]),
                    "candidate_ids": list(value.get("candidate_ids", [])),
                    "evidence_image_indices": list(value.get("evidence_image_indices", [])),
                }
                for cid, value in normalized.items()
            }
            panorama = ROOT / str(prefix["current_panorama_path"])
            run_dir = panorama.parents[1]
            causal = {
                int(item["step"]): item
                for item in read_jsonl(run_dir / "causal_prefix_records.jsonl")
            }
            if step not in causal:
                raise ProxyREEError("causal array reference is missing")
            arrays = ROOT / str(causal[step]["arrays"]["path"])
            if sha256_file(arrays) != causal[step]["arrays"]["sha256"]:
                raise ProxyREEError("causal array hash drift")
            arrays_paths.add(arrays)
            prefix_sources.append({
                "step": step,
                "candidate_ids": list(prefix["candidate_ids"]),
                "arrays": inventory(arrays),
                "qwen_evidence": inventory(evidence_path),
            })

        roles = infer_proxy_dec_roles(graph, factors_by_step)
        constraints = {}
        for graph_row in graph:
            cid = str(graph_row["constraint_id"])
            role = roles[cid]
            factors = []
            for step in sorted(factors_by_step):
                source = factors_by_step[step][cid]
                factors.append({
                    "step": step,
                    "instantiated": source["instantiated"] if role in DEC_ROLES else None,
                    "distinguishable": source["distinguishable"] if role in DEC_ROLES else None,
                    "resolved": source["resolved"] if role in DEC_ROLES else None,
                })
            constraints[cid] = {"dec_role": role, "factor_by_step": factors}

        row = {
            "schema_version": "revealnav-mf3zp-codex-proxy-label/1",
            "revision": REVISION,
            "provenance": {
                "label_class": "AI_PROXY_NOT_HUMAN_NOT_GOLD",
                "factor_source": "cached_outcome_blind_qwen3.8-max",
                "dec_source": "deterministic_dependency_evidence_rule",
                "human_verified": False,
                "gold": False,
                "qwen_api_calls_this_revision": 0,
            },
            "event_id": event_id,
            "dataset": str(template["dataset"]),
            "scene_id": str(template["scene_id"]),
            "episode_id": str(template["episode_id"]),
            "instruction": str(template["instruction"]),
            "decision_step": int(template["decision_step"]),
            "constraint_graph_sha256": str(template["constraint_graph_sha256"]),
            "constraint_graph": graph,
            "constraints": constraints,
            "prefix_sources": prefix_sources,
            "public_split_access": PUBLIC_CLOSED,
        }
        _reject_outcomes(row)
        rows.append(row)

    atomic_jsonl(LABELS, rows)
    role_counts = Counter(
        item["dec_role"] for row in rows for item in row["constraints"].values()
    )
    manifest = {
        "schema_version": "revealnav-mf3zp-codex-proxy-label-manifest/1",
        "revision": REVISION,
        "status": "AI_PROXY_LABELS_COMPLETE_NOT_HUMAN_NOT_GOLD",
        "events": len(rows),
        "domains": Counter(str(row["dataset"]) for row in rows),
        "scenes": len({str(row["scene_id"]) for row in rows}),
        "prefixes": sum(len(row["prefix_sources"]) for row in rows),
        "role_counts": dict(sorted(role_counts.items())),
        "labels": inventory(LABELS),
        "source_evidence_inventory": {
            "count": len(source_evidence_paths),
            "sha256": stable_sha256([inventory(path) for path in sorted(source_evidence_paths)]),
        },
        "causal_array_inventory": {
            "count": len(arrays_paths),
            "sha256": stable_sha256([inventory(path) for path in sorted(arrays_paths)]),
        },
        "human_labels_fabricated": False,
        "formal_label_validity_pass": False,
        "qwen_api_calls": 0,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    atomic_json(LABEL_MANIFEST, manifest)
    protocol = {
        "schema_version": "revealnav-mf3zp-codex-proxy-ree-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_CODEX_PROXY_REE_RESULT",
        "purpose": "exploratory machine-label temporal learnability only",
        "inputs": {
            "selection": inventory(SELECTION),
            "blank_human_template": inventory(REVIEW_TEMPLATE),
            "pilot_events": inventory(PILOT_EVENTS),
            "source_requests": inventory(SOURCE_REQUESTS),
            "proxy_labels": inventory(LABELS),
            "proxy_manifest": inventory(LABEL_MANIFEST),
        },
        "training": {
            "folds": FOLDS,
            "raw_scene_disjoint": True,
            "arms": ["snapshot", "temporal"],
            "hidden_dim": 64,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "prediction_boundary": 0.5,
            "hyperparameter_search": False,
            "expiry_supervision": "UNAVAILABLE_NOT_FABRICATED",
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
    atomic_json(PROTOCOL, protocol)
    return manifest


def verify_proxy_protocol() -> dict[str, object]:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "SEALED_BEFORE_CODEX_PROXY_REE_RESULT":
        raise ProxyREEError("proxy protocol status drift")
    for key, value in protocol["inputs"].items():
        path = ROOT / str(value["path"])
        if inventory(path) != value:
            raise ProxyREEError(f"proxy protocol input drift: {key}")
    if any(protocol.get("authorization", {}).values()):
        raise ProxyREEError("proxy protocol opened downstream authorization")
    if protocol.get("public_split_access") != PUBLIC_CLOSED:
        raise ProxyREEError("proxy protocol opened a public split")
    return protocol


def _fixed_projection(input_dim: int, output_dim: int) -> np.ndarray:
    row = np.arange(1, input_dim + 1, dtype=np.float64)[:, None]
    column = np.arange(1, output_dim + 1, dtype=np.float64)[None, :]
    value = np.sin(row * column * 0.017) + np.cos(row * (column + 1.0) * 0.013)
    value /= np.sqrt(np.square(value).sum(axis=0, keepdims=True)).clip(1e-12)
    return value.astype(np.float32)


_EMBED_PROJECTION = _fixed_projection(768, PROJECTION_DIM)


def _normalized_projection(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (768,) or not np.isfinite(value).all():
        raise ProxyREEError("invalid frozen embedding")
    norm = float(np.linalg.norm(value))
    return (value / max(norm, 1e-12)) @ _EMBED_PROJECTION


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _constraint_vector(graph_row: Mapping[str, object]) -> np.ndarray:
    text = " ".join(
        str(graph_row.get(key) or "")
        for key in ("kind", "subject", "relation", "object")
    ).casefold()
    tokens = re.findall(r"[a-z0-9]+", text)
    result = np.zeros(CONSTRAINT_DIM, dtype=np.float32)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % CONSTRAINT_DIM
        result[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(result))
    return result / max(norm, 1.0)


def causal_feature_rows(
    prefix_sources: Sequence[Mapping[str, object]],
    graph_row: Mapping[str, object],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    previous_checkpoint: np.ndarray | None = None
    previous_candidates: set[str] = set()
    constraint = _constraint_vector(graph_row)
    for prefix in prefix_sources:
        path = ROOT / str(prefix["arrays"]["path"])
        if sha256_file(path) != prefix["arrays"]["sha256"]:
            raise ProxyREEError("training array hash drift")
        with np.load(path, allow_pickle=False) as data:
            instruction = np.asarray(data["instruction"], dtype=np.float32)
            checkpoint = np.asarray(data["checkpoint"], dtype=np.float32)
            actions = np.asarray(data["action_embeddings"], dtype=np.float32)
            scores = np.asarray(data["policy_scores"], dtype=np.float32)
            positions = np.asarray(data["position_features"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1:] != (768,) or len(actions) != len(scores):
            raise ProxyREEError("candidate feature alignment drift")
        mean_action = actions.mean(axis=0) if len(actions) else np.zeros(768, dtype=np.float32)
        top_action = actions[int(np.argmax(scores))] if len(actions) else mean_action
        score_sorted = np.sort(scores)
        score_summary = np.asarray([
            float(len(scores)), float(scores.max()) if len(scores) else 0.0,
            float(scores.mean()) if len(scores) else 0.0,
            float(scores.std()) if len(scores) else 0.0,
            float(scores.min()) if len(scores) else 0.0,
            float(score_sorted[-1] - score_sorted[-2]) if len(scores) > 1 else 0.0,
        ], dtype=np.float32)
        if len(positions):
            position_summary = np.concatenate((positions.mean(axis=0), positions.std(axis=0))).astype(np.float32)
        else:
            position_summary = np.zeros(14, dtype=np.float32)
        candidates = {str(value) for value in prefix["candidate_ids"]}
        union = candidates | previous_candidates
        jaccard = len(candidates & previous_candidates) / len(union) if union else 1.0
        drift_cos = _cosine(previous_checkpoint, checkpoint) if previous_checkpoint is not None else 1.0
        drift_l2 = float(np.linalg.norm(checkpoint - previous_checkpoint)) if previous_checkpoint is not None else 0.0
        relations = np.asarray([
            float(prefix["step"]),
            _cosine(instruction, checkpoint),
            _cosine(instruction, mean_action),
            _cosine(checkpoint, mean_action),
            drift_cos, drift_l2, jaccard,
            float(len(candidates - previous_candidates)),
            float(len(previous_candidates - candidates)),
        ], dtype=np.float32)
        row = np.concatenate((
            _normalized_projection(instruction),
            _normalized_projection(checkpoint),
            _normalized_projection(mean_action),
            _normalized_projection(top_action),
            constraint, score_summary, position_summary, relations,
        )).astype(np.float32)
        if not np.isfinite(row).all():
            raise ProxyREEError("non-finite causal proxy feature")
        rows.append(row)
        previous_checkpoint = checkpoint
        previous_candidates = candidates
    return np.stack(rows)


@dataclass(frozen=True)
class ProxyExample:
    event_id: str
    dataset: str
    scene_id: str
    constraint_id: str
    features: np.ndarray
    targets: np.ndarray


def load_proxy_examples() -> list[ProxyExample]:
    verify_proxy_protocol()
    examples: list[ProxyExample] = []
    for row in read_jsonl(LABELS):
        graph = {str(item["constraint_id"]): item for item in row["constraint_graph"]}
        for cid, item in row["constraints"].items():
            if item["dec_role"] not in DEC_ROLES:
                continue
            targets = np.asarray([
                [factor["instantiated"], factor["distinguishable"], factor["resolved"]]
                for factor in item["factor_by_step"]
            ], dtype=np.float32)
            features = causal_feature_rows(row["prefix_sources"], graph[str(cid)])
            if targets.shape != (len(features), 3):
                raise ProxyREEError("proxy feature/target length mismatch")
            examples.append(ProxyExample(
                event_id=str(row["event_id"]), dataset=str(row["dataset"]),
                scene_id=str(row["scene_id"]), constraint_id=str(cid),
                features=features, targets=targets,
            ))
    if not examples:
        raise ProxyREEError("proxy label set contains no DEC examples")
    return examples


def scene_fold(scene_id: str) -> int:
    digest = hashlib.sha256(f"mf3zp-codex-proxy-ree-fold-v1:{scene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _pad_examples(examples: Sequence[ProxyExample]) -> tuple[Tensor, Tensor, Tensor]:
    width = examples[0].features.shape[1]
    length = max(len(example.features) for example in examples)
    features = np.zeros((len(examples), length, width), dtype=np.float32)
    targets = np.zeros((len(examples), length, 3), dtype=np.float32)
    mask = np.zeros((len(examples), length), dtype=bool)
    for index, example in enumerate(examples):
        n = len(example.features)
        features[index, :n] = example.features
        targets[index, :n] = example.targets
        mask[index, :n] = True
    return torch.from_numpy(features), torch.from_numpy(targets), torch.from_numpy(mask)


def _standardize(
    train: Sequence[ProxyExample], evaluate: Sequence[ProxyExample],
) -> tuple[list[ProxyExample], list[ProxyExample]]:
    values = np.concatenate([example.features for example in train], axis=0)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0

    def apply(examples: Sequence[ProxyExample]) -> list[ProxyExample]:
        return [ProxyExample(
            event_id=value.event_id, dataset=value.dataset, scene_id=value.scene_id,
            constraint_id=value.constraint_id,
            features=((value.features - mean) / scale).astype(np.float32),
            targets=value.targets,
        ) for value in examples]
    return apply(train), apply(evaluate)


def _model_loss(model: TemporalRevealExpiryEncoder, features: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    output = model(features, mask)
    logits = torch.stack((
        output.target_in_set_logit,
        output.separation_logit,
        output.evidence_logit,
    ), dim=-1)
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def _snapshot_examples(examples: Sequence[ProxyExample]) -> tuple[list[ProxyExample], list[tuple[int, int]]]:
    result = []
    positions = []
    for parent, example in enumerate(examples):
        for step in range(len(example.features)):
            result.append(ProxyExample(
                event_id=example.event_id, dataset=example.dataset,
                scene_id=example.scene_id, constraint_id=example.constraint_id,
                features=example.features[step:step + 1],
                targets=example.targets[step:step + 1],
            ))
            positions.append((parent, step))
    return result, positions


def _fit_predict_fold(
    train: Sequence[ProxyExample],
    evaluate: Sequence[ProxyExample],
    *,
    arm: str,
    fold: int,
    device: torch.device,
) -> tuple[list[np.ndarray], float]:
    train, evaluate = _standardize(train, evaluate)
    positions: list[tuple[int, int]] | None = None
    original_evaluate = evaluate
    if arm == "snapshot":
        train, _ = _snapshot_examples(train)
        evaluate, positions = _snapshot_examples(evaluate)
    elif arm != "temporal":
        raise ProxyREEError("unknown proxy arm")
    train_x, train_y, train_mask = _pad_examples(train)
    eval_x, _, eval_mask = _pad_examples(evaluate)
    torch.manual_seed(SEED + fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + fold)
    model = TemporalRevealExpiryEncoder(input_dim=train_x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_x, train_y, train_mask = train_x.to(device), train_y.to(device), train_mask.to(device)
    model.train()
    final_loss = float("nan")
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = _model_loss(model, train_x, train_y, train_mask)
        if not bool(torch.isfinite(loss)):
            raise ProxyREEError("proxy training loss became non-finite")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    with torch.no_grad():
        output = model(eval_x.to(device), eval_mask.to(device))
        probabilities = torch.sigmoid(torch.stack((
            output.target_in_set_logit,
            output.separation_logit,
            output.evidence_logit,
        ), dim=-1)).cpu().numpy()
    if arm == "temporal":
        predictions = [probabilities[index, :len(example.features)] for index, example in enumerate(evaluate)]
    else:
        predictions = [np.zeros_like(example.targets) for example in original_evaluate]
        assert positions is not None
        for flat_index, (parent, step) in enumerate(positions):
            predictions[parent][step] = probabilities[flat_index, 0]
    return predictions, final_loss


def _binary_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = probability >= 0.5
    eps = 1e-7
    nll = -np.mean(truth * np.log(np.clip(probability, eps, 1 - eps)) + (1 - truth) * np.log(np.clip(1 - probability, eps, 1 - eps)))
    tp = int(np.sum((truth == 1) & prediction))
    fp = int(np.sum((truth == 0) & prediction))
    fn = int(np.sum((truth == 1) & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": int(len(truth)), "positive_count": int(np.sum(truth)),
        "accuracy": float(np.mean(prediction == truth)), "f1_positive": float(f1),
        "nll": float(nll),
    }


def _macro_f1(truth: Sequence[str], prediction: Sequence[str]) -> float:
    categories = ("U", "A", "D")
    values = []
    for category in categories:
        tp = sum(a == category and b == category for a, b in zip(truth, prediction, strict=True))
        fp = sum(a != category and b == category for a, b in zip(truth, prediction, strict=True))
        fn = sum(a == category and b != category for a, b in zip(truth, prediction, strict=True))
        if tp + fp + fn == 0:
            continue
        values.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values)) if values else float("nan")


def _arm_metrics(
    examples: Sequence[ProxyExample], predictions: Sequence[np.ndarray], domain: str,
) -> dict[str, object]:
    chosen = [index for index, example in enumerate(examples) if example.dataset == domain]
    truth = np.concatenate([examples[index].targets for index in chosen], axis=0)
    probability = np.concatenate([predictions[index] for index in chosen], axis=0)
    factors = {
        name: _binary_metrics(truth[:, column], probability[:, column])
        for column, name in enumerate(("instantiated", "distinguishable", "resolved"))
    }
    truth_uad: list[str] = []
    predicted_uad: list[str] = []
    for index in chosen:
        actual = examples[index].targets.astype(bool)
        predicted = predictions[index] >= 0.5
        truth_uad.extend(value.value for value in derive_constraint_uad(actual[:, 0], actual[:, 1], actual[:, 2]))
        predicted_uad.extend(value.value for value in derive_constraint_uad(predicted[:, 0], predicted[:, 1], predicted[:, 2]))
    return {
        "factor_mean_nll": float(np.mean([value["nll"] for value in factors.values()])),
        "factor_mean_accuracy": float(np.mean([value["accuracy"] for value in factors.values()])),
        "uad_macro_f1": _macro_f1(truth_uad, predicted_uad),
        "uad_counts": dict(Counter(truth_uad)),
        "factors": factors,
    }


def train_proxy_ree(device_name: str) -> dict[str, object]:
    protocol = verify_proxy_protocol()
    examples = load_proxy_examples()
    scenes = {example.scene_id for example in examples}
    fold_scenes = {fold: sorted(scene for scene in scenes if scene_fold(scene) == fold) for fold in range(FOLDS)}
    if any(not value for value in fold_scenes.values()):
        raise ProxyREEError("proxy scene fold is empty")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProxyREEError("CUDA device requested but unavailable")
    predictions = {
        "snapshot": [None] * len(examples),
        "temporal": [None] * len(examples),
    }
    fold_reports = []
    for fold in range(FOLDS):
        held = set(fold_scenes[fold])
        train_indices = [index for index, value in enumerate(examples) if value.scene_id not in held]
        eval_indices = [index for index, value in enumerate(examples) if value.scene_id in held]
        if not train_indices or not eval_indices or {examples[index].scene_id for index in train_indices} & held:
            raise ProxyREEError("scene-disjoint fold construction failed")
        report = {"fold": fold, "held_scenes": sorted(held), "train_examples": len(train_indices), "evaluate_examples": len(eval_indices), "arms": {}}
        for arm in ("snapshot", "temporal"):
            values, loss = _fit_predict_fold(
                [examples[index] for index in train_indices],
                [examples[index] for index in eval_indices],
                arm=arm, fold=fold, device=device,
            )
            for index, value in zip(eval_indices, values, strict=True):
                predictions[arm][index] = value
            report["arms"][arm] = {"final_training_loss": loss}
        fold_reports.append(report)
    if any(value is None for arm in predictions.values() for value in arm):
        raise ProxyREEError("OOF prediction is incomplete")

    metrics = {}
    positive_domains = []
    for domain in ("R2R", "RxR"):
        snapshot = _arm_metrics(examples, predictions["snapshot"], domain)  # type: ignore[arg-type]
        temporal = _arm_metrics(examples, predictions["temporal"], domain)  # type: ignore[arg-type]
        delta_nll = float(snapshot["factor_mean_nll"] - temporal["factor_mean_nll"])
        delta_uad = float(temporal["uad_macro_f1"] - snapshot["uad_macro_f1"])
        positive = delta_nll > 0.0 and delta_uad > 0.0
        positive_domains.append(positive)
        metrics[domain] = {
            "snapshot": snapshot,
            "temporal": temporal,
            "temporal_minus_snapshot": {
                "factor_nll_improvement": delta_nll,
                "uad_macro_f1_improvement": delta_uad,
            },
            "proxy_temporal_signal_positive": positive,
        }
    signal = (
        "EXPLORATORY_PROXY_TEMPORAL_SIGNAL_POSITIVE"
        if all(positive_domains)
        else "EXPLORATORY_PROXY_TEMPORAL_SIGNAL_MIXED_OR_NEGATIVE"
    )
    result = {
        "schema_version": "revealnav-mf3zp-codex-proxy-ree-result/1",
        "revision": REVISION,
        "status": "EXPLORATORY_CODEX_PROXY_REE_COMPLETE",
        "signal_status": signal,
        "protocol_sha256": sha256_file(PROTOCOL),
        "label_manifest_sha256": sha256_file(LABEL_MANIFEST),
        "events": len({example.event_id for example in examples}),
        "constraint_sequences": len(examples),
        "scenes": len(scenes),
        "domains": dict(Counter(example.dataset for example in examples)),
        "feature_dim": int(examples[0].features.shape[1]),
        "folds": fold_reports,
        "metrics": metrics,
        "limitations": [
            "S/G/E targets are cached provisional Qwen labels, not human labels",
            "DEC roles are deterministic AI-proxy roles, not expert DEC",
            "expiry supervision is unavailable and was not fabricated",
            "this result cannot authorize Oracle Headroom, formal REE, skill policy, or public evaluation",
        ],
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
    "ProxyREEError", "build_proxy_labels", "infer_proxy_dec_roles",
    "load_proxy_examples", "scene_fold", "train_proxy_ree",
]
