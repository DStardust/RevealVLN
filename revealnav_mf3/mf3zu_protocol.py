"""Sealed RxR-only evidence-memory feasibility protocol for MF3ZU.

MF3ZU is a new diagnostic revision.  It deliberately leaves the failed,
dual-domain MF3ZT revision untouched.  Exact target existence is used only to
fix legal candidate-ranking support.  A separate sanitized population prevents
annotation or memory-required classification from seeing target identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zu_rxr_evidence_memory_feasibility_v1"
OUTPUT = ROOT / "artifacts" / "training" / REVISION
PROTOCOL_PATH = OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PROTOCOL.json"
POPULATION_PATH = OUTPUT / "MF3ZU_RXR_DECISION_POPULATION.jsonl"
POPULATION_MANIFEST_PATH = OUTPUT / "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json"
EXACT_TARGETS_PATH = OUTPUT / "MF3ZU_RXR_EXACT_TARGETS.jsonl"
EVIDENCE_MEMORY_PATH = OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl"
EVIDENCE_MEMORY_MANIFEST_PATH = OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json"
RESULT_PATH = OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_RESULT.json"

# The reviewed main revision containing the immutable MF3ZT support result.
BASE_COMMIT = "d2c08cc068c3fc53aecf351ee4fbd50db6f5d9d4"
DATASET = "RxR"
PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}

FOLDS = 5
FOLD_SALT = "mf3zu-rxr-evidence-memory-feasibility-v1-scene-folds"
MIN_CANDIDATES = 2
EXPECTED_SOURCE_ROWS = 1_920
EXPECTED_CANDIDATE_ELIGIBLE_ROWS = 1_815
EXPECTED_POPULATION_ROWS = 1_428
EXPECTED_SOURCE_EPISODES = 156
EXPECTED_SOURCE_SCENES = 59
EXPECTED_POPULATION_EPISODES = 154
EXPECTED_POPULATION_SCENES = 59

EVIDENCE_ONTOLOGY = (
    "LANDMARK_SEEN",
    "LANDMARK_PASSED",
    "RELATION_SATISFIED",
    "ORDINAL_COUNT",
    "DIRECTIONAL_CONTEXT",
)
CONFIDENCE_CLASSES = ("OBSERVED", "AMBIGUOUS", "ABSENT")
ARMS = (
    "ETP_CURRENT",
    "ETP_PLUS_EVIDENCE_MEMORY",
    "ETP_PLUS_SHUFFLED_MEMORY",
)
K_MEM = 8
EVIDENCE_RECORD_DIM = 77
CANDIDATE_BINDING_DIM = 1
EVIDENCE_FEATURE_DIM = EVIDENCE_RECORD_DIM + CANDIDATE_BINDING_DIM
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_901
TRAINING_SEED = 20_260_901
MIN_MEMORY_REQUIRED_DECISIONS = 50
MIN_MEMORY_REQUIRED_SCENES = 10

QWEN_EXTRACTOR = {
    "model": "qwen3.8-max",
    "temperature": 0,
    "thinking": False,
    "max_tokens": 8_000,
    "model_sweep": False,
    "prompt_search": False,
    "ensemble": False,
}

MF3B_TARGET_PROTOCOL = (
    ROOT
    / "artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_PROTOCOL.json"
)
MF3B_TARGET_MANIFEST = (
    ROOT
    / "artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
)
RXR_TRAIN_GUIDE = (
    ROOT
    / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
RXR_ETP_CHECKPOINT = (
    ROOT
    / "third_party/ETP-R1/data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
ETP_BACKBONE = (
    ROOT
    / "third_party/ETP-R1/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
OBSERVATION_REPLAY_WORKER = ROOT / "scripts/mf3zp_observation_worker_v2.py"
QWEN_CLIENT = ROOT / "revealnav_mf3/qwen_evidence_annotation.py"
MF3ZT_METHOD = ROOT / "METHOD_REVISION_3ZT_EVIDENCE_MEMORY_DECISION_PROBE.md"
MF3ZT_PROTOCOL = (
    ROOT
    / "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json"
)
MF3ZT_TARGET_AUDIT = (
    ROOT
    / "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_DECISION_TARGET_SUPPORT_AUDIT.json"
)
MF3ZT_RESULT = (
    ROOT
    / "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_RESULT.json"
)

EXPECTED_SOURCE_SHA256 = {
    "mf3b_target_protocol": "c857b0863062074987d497bb1b5b4a3cfbf768d1396ff15308e52ec6583e7346",
    "mf3b_target_manifest": "36884bae31718bb859f5856103654f5b3f25979fdfd0319c1ca344f00328e034",
    "rxr_train_guide": "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    "rxr_etp_checkpoint": "3796c9c94ff8674b8cfe99f2b4aab0f4b391f0d4c9c1e167e4736b3848f27821",
    "etp_backbone": "203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf",
    "observation_replay_worker": "162fba817bc36f30823dfeb3afc713b7e52354b6c643bb244f224e2fbb992124",
    "qwen_client": "a84b6502693caba9e8139b017d9f1b588d7d8b1c51ec5ae7889ba9fb4a866c07",
    "mf3zt_method": "3a962ee53567d79d58dfca049c24374e3d5b35a3891c02af42ff168cb11c12c2",
    "mf3zt_protocol": "71ba83a89b58eb7797a953cbdae8b03d51dd4fdae7b6618c451a511d7fd01af2",
    "mf3zt_target_audit": "31b10d600e8bce5a1c82c82481b146ed75489de287ac2d049c43508b0d6a958b",
    "mf3zt_result": "0910c61b9cf5d45e845447bf60a1df7901d05aa62f3dfd37b3090429712b608e",
}

SOURCE_PATHS = {
    "mf3b_target_protocol": MF3B_TARGET_PROTOCOL,
    "mf3b_target_manifest": MF3B_TARGET_MANIFEST,
    "rxr_train_guide": RXR_TRAIN_GUIDE,
    "rxr_etp_checkpoint": RXR_ETP_CHECKPOINT,
    "etp_backbone": ETP_BACKBONE,
    "observation_replay_worker": OBSERVATION_REPLAY_WORKER,
    "qwen_client": QWEN_CLIENT,
    "mf3zt_method": MF3ZT_METHOD,
    "mf3zt_protocol": MF3ZT_PROTOCOL,
    "mf3zt_target_audit": MF3ZT_TARGET_AUDIT,
    "mf3zt_result": MF3ZT_RESULT,
}

IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY.md",
    "revealnav_mf3/mf3zu_protocol.py",
    "revealnav_mf3/mf3zu_evidence_memory.py",
    "revealnav_mf3/mf3zu_evidence_memory_reranker.py",
    "revealnav_mf3/mf3zu_evidence_memory_metrics.py",
    "scripts/build_mf3zu_rxr_decision_population.py",
    "scripts/seal_mf3zu_protocol.py",
    "scripts/collect_mf3zu_rxr_observations.py",
    "scripts/annotate_mf3zu_rxr_evidence.py",
    "scripts/build_mf3zu_evidence_memory.py",
    "scripts/train_mf3zu_rxr_feasibility.py",
    "scripts/audit_mf3zu_rxr_feasibility_result.py",
    "scripts/run_mf3zu_rxr_feasibility_pipeline.py",
)

FORBIDDEN_POPULATION_FIELDS = frozenset(
    {
        "target",
        "target_index",
        "teacher_action_id_label_only",
        "teacher_action_index_label_only",
        "correct_candidate",
        "success",
        "reward",
        "utility",
        "future_observation",
    }
)


class ProtocolError(RuntimeError):
    """Raised when a sealed MF3ZU boundary is missing or drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_local_file(path: Path, root: Path = ROOT) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"missing or unsafe file: {path}")
    if resolved != root and root not in resolved.parents:
        raise ProtocolError(f"file is outside declared source root: {path}")
    return resolved


def inventory(
    path: Path,
    *,
    expected_sha256: str | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    resolved = _safe_local_file(path, root)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"source hash drift: {resolved}")
    return {
        "path": str(resolved.relative_to(root.resolve())),
        "bytes": int(resolved.stat().st_size),
        "sha256": digest,
    }


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ensure_base_commit() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProtocolError("MF3ZU reviewed base commit is not an ancestor of HEAD")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def scene_fold_mapping(scene_ids: Iterable[str]) -> dict[str, int]:
    """Assign hashed scenes round-robin to five folds.

    Hashing determines a stable pseudorandom order while round-robin assignment
    keeps scene counts balanced to within one.  Dataset identity is deliberately
    absent from the key because the clustering unit is the raw MP3D scene.
    """

    scenes = {str(scene_id) for scene_id in scene_ids}
    if not scenes or "" in scenes:
        raise ProtocolError("scene IDs must be non-empty")

    def key(scene_id: str) -> tuple[str, str]:
        payload = f"{FOLD_SALT}\0{scene_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), scene_id

    ordered = sorted(scenes, key=key)
    return {scene_id: index % FOLDS for index, scene_id in enumerate(ordered)}


def _resolve_record_file(source_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProtocolError("manifest record has no feature path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ProtocolError("manifest feature path must be relative")
    return _safe_local_file(source_root / candidate, source_root)


def _load_shadow_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ProtocolError(f"malformed shadow row {path}:{line_number}")
        rows.append(value)
    return rows


def _population_row_has_forbidden_field(row: Mapping[str, object]) -> bool:
    return any(field in row for field in FORBIDDEN_POPULATION_FIELDS)


def build_population_rows(
    manifest_path: Path = MF3B_TARGET_MANIFEST,
    *,
    source_root: Path = ROOT,
    enforce_frozen_counts: bool = True,
    verify_feature_hashes: bool = True,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Build exact-support and sanitized RxR feasibility artifacts in memory.

    ``target_index`` participates only in the fixed legal-support predicate.  A
    separate exact-target table retains the selected slot for later training;
    the sanitized population passed to replay/annotation contains no target
    field or value.
    """

    # NumPy is intentionally local: importing the protocol itself does not need
    # the training environment.
    import numpy as np

    manifest = _read_object(_safe_local_file(manifest_path, source_root))
    if manifest.get("public_unseen_authorized") is not False:
        raise ProtocolError("MF3B source authorizes public unseen")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ProtocolError("MF3B manifest has no records")

    prepared: list[tuple[dict[str, object], Path, list[int], list[dict[str, object]]]] = []
    source_row_count = 0
    scenes: list[str] = []
    source_episodes: set[str] = set()

    for record in records:
        if not isinstance(record, dict):
            raise ProtocolError("malformed MF3B manifest record")
        scene_id = str(record.get("scene_id", ""))
        episode_id = str(record.get("episode_id", ""))
        if not scene_id or not episode_id:
            raise ProtocolError("source record lacks scene or episode identity")
        feature_path = _resolve_record_file(source_root, record.get("path"))
        if verify_feature_hashes:
            expected = record.get("sha256")
            if not isinstance(expected, str) or sha256_file(feature_path) != expected:
                raise ProtocolError(f"MF3B feature hash drift: {feature_path}")

        with np.load(feature_path, allow_pickle=False) as arrays:
            required = {"candidate_mask", "target_index", "target_in_set"}
            if not required.issubset(arrays.files):
                raise ProtocolError(f"exact support arrays missing: {feature_path}")
            mask = np.asarray(arrays["candidate_mask"])
            target_index = np.asarray(arrays["target_index"])
            target_in_set = np.asarray(arrays["target_in_set"])
        if mask.ndim != 2 or mask.dtype.kind != "b":
            raise ProtocolError(f"candidate_mask must be a 2-D boolean array: {feature_path}")
        if target_index.shape != (mask.shape[0],) or target_index.dtype.kind not in "iu":
            raise ProtocolError(f"target_index shape or dtype drift: {feature_path}")
        if target_in_set.shape != (mask.shape[0],):
            raise ProtocolError(f"target_in_set shape drift: {feature_path}")
        source_row_count += int(mask.shape[0])
        eligible = [
            int(index)
            for index, candidate_row in enumerate(mask)
            if int(np.count_nonzero(candidate_row)) >= MIN_CANDIDATES
        ]

        shadow_path = _safe_local_file(feature_path.parent / "uad_shadow.jsonl", source_root)
        shadow_rows = _load_shadow_rows(shadow_path)
        if len(shadow_rows) != len(eligible):
            raise ProtocolError(
                f"eligible-feature/shadow row mismatch for episode {episode_id}"
            )
        prepared.append((record, feature_path, eligible, shadow_rows))
        scenes.append(scene_id)
        source_episodes.add(episode_id)

    folds = scene_fold_mapping(scenes)
    population: list[dict[str, object]] = []
    exact_targets: list[dict[str, object]] = []
    candidate_eligible_count = 0
    selected_feature_physical_mismatch_count = 0
    for record, feature_path, eligible, shadow_rows in prepared:
        scene_id = str(record["scene_id"])
        episode_id = str(record["episode_id"])
        shadow_path = feature_path.parent / "uad_shadow.jsonl"
        native_trace_path = _safe_local_file(feature_path.parent / "base_trace.jsonl", source_root)
        feature_relative = str(feature_path.resolve().relative_to(source_root.resolve()))
        shadow_relative = str(shadow_path.resolve().relative_to(source_root.resolve()))
        trace_relative = str(native_trace_path.resolve().relative_to(source_root.resolve()))

        native_trace_rows = _load_shadow_rows(native_trace_path)
        trace_by_physical_step: dict[int, tuple[int, dict[str, object]]] = {}
        for trace_row_index, trace_row in enumerate(native_trace_rows):
            physical_step = trace_row.get("i")
            if not isinstance(physical_step, int) or physical_step < 0:
                raise ProtocolError("native trace has invalid physical step")
            if physical_step in trace_by_physical_step:
                raise ProtocolError("native trace repeats a physical step")
            trace_by_physical_step[physical_step] = (trace_row_index, trace_row)

        expected_previous = "0" * 64
        previous_physical_step = -1
        for shadow in shadow_rows:
            if shadow.get("previous_hash") != expected_previous:
                raise ProtocolError("shadow decision hash chain drift")
            record_hash = shadow.get("record_hash")
            if not isinstance(record_hash, str) or len(record_hash) != 64:
                raise ProtocolError("shadow decision record hash drift")
            expected_previous = record_hash
            physical_step = shadow.get("step")
            if (
                not isinstance(physical_step, int)
                or physical_step <= previous_physical_step
                or physical_step not in trace_by_physical_step
            ):
                raise ProtocolError("shadow physical-step mapping drift")
            previous_physical_step = physical_step

        with np.load(feature_path, allow_pickle=False) as arrays:
            mask = np.asarray(arrays["candidate_mask"])
            target_index = np.asarray(arrays["target_index"])
            target_in_set = np.asarray(arrays["target_in_set"])

        for eligible_ordinal, (feature_row, shadow) in enumerate(
            zip(eligible, shadow_rows, strict=True)
        ):
            candidate_eligible_count += 1
            if shadow.get("public_unseen_authorized") is not False:
                raise ProtocolError("shadow row opens public unseen")
            if shadow.get("teacher_used_as_online_input") is not False:
                raise ProtocolError("teacher was used as an online feature")
            action_ids = shadow.get("current_local_action_ids")
            action_indices = shadow.get("current_local_action_indices")
            if not isinstance(action_ids, list) or not isinstance(action_indices, list):
                raise ProtocolError("shadow row lacks current candidate identity")
            feature_slots = np.flatnonzero(mask[feature_row]).astype(int).tolist()
            if not (
                len(action_ids)
                == len(action_indices)
                == len(feature_slots)
                >= MIN_CANDIDATES
            ):
                raise ProtocolError("candidate identity/count drift")
            if len({str(value) for value in action_ids}) != len(action_ids):
                raise ProtocolError("duplicate candidate action ID")
            physical_step = shadow.get("step")
            if not isinstance(physical_step, int) or physical_step < 0:
                raise ProtocolError("invalid causal decision step")
            trace_row_index, trace_row = trace_by_physical_step[physical_step]
            if trace_row.get("i") != physical_step:
                raise ProtocolError("native trace physical-step resolution failed")

            target_slot = int(target_index[feature_row])
            exact_rankable = (
                0 <= target_slot < mask.shape[1]
                and bool(mask[feature_row, target_slot])
            )
            if exact_rankable != bool(float(target_in_set[feature_row]) > 0.5):
                raise ProtocolError("target existence/index support arrays disagree")
            if not exact_rankable:
                continue
            if feature_row != physical_step:
                selected_feature_physical_mismatch_count += 1

            event_id = f"RxR:{scene_id}:{episode_id}:{physical_step}"
            row = {
                "schema_version": "revealnav-mf3zu-rxr-feasibility-population-row/1",
                "revision": REVISION,
                "event_id": event_id,
                "dataset": DATASET,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "decision_step": physical_step,
                "physical_decision_step": physical_step,
                "feature_row_index": feature_row,
                "eligible_decision_ordinal": eligible_ordinal,
                "native_trace_row_index": trace_row_index,
                "feature_row_equals_physical_step_assumed": False,
                "feature_shadow_mapping": "ascending_eligible_feature_ordinal_to_hash_chained_shadow_decision_ordinal",
                "scene_fold": folds[scene_id],
                "candidate_count": len(action_ids),
                "candidate_action_ids": [str(value) for value in action_ids],
                "candidate_graph_indices": [int(value) for value in action_indices],
                "active_candidate_feature_slots": feature_slots,
                "candidate_coordinate_binding_status": "UNBOUND_UNTIL_REPLAY_EMBEDDING_AND_SCORE_BYTE_MATCH",
                "native_action_id": (
                    None
                    if shadow.get("native_action_id") is None
                    else str(shadow["native_action_id"])
                ),
                "source_feature_path": feature_relative,
                "source_feature_sha256": str(record["sha256"]),
                "source_shadow_path": shadow_relative,
                "source_shadow_sha256": sha256_file(shadow_path),
                "source_shadow_record_hash": str(shadow.get("record_hash", "")),
                "source_native_trace_path": trace_relative,
                "source_native_trace_sha256": sha256_file(native_trace_path),
                "population_selection_rule": "candidate_mask_count>=2_and_exact_target_feature_slot_active",
                "exact_support_status": "SEALED_IN_SEPARATE_TARGET_ARTIFACT",
                "causal_replay_end_step": physical_step,
                "memory_required_label": "NOT_YET_MATERIALIZED",
                "evidence_memory_status": "NOT_YET_MATERIALIZED",
            }
            if _population_row_has_forbidden_field(row):
                raise ProtocolError("candidate target leaked into population row")
            population.append(row)
            exact_targets.append(
                {
                    "schema_version": "revealnav-mf3zu-rxr-exact-target-row/1",
                    "revision": REVISION,
                    "event_id": event_id,
                    "target_index": target_slot,
                    "target_feature_slot": target_slot,
                    "coordinate_system": "MF3B_candidate_feature_slot",
                    "source_feature_path": feature_relative,
                    "source_feature_sha256": str(record["sha256"]),
                    "source_feature_row_index": feature_row,
                    "source_array": "target_index",
                    "support_rule": "target_index_in_bounds_and_candidate_mask_true",
                    "baseline_score_or_correctness_used": False,
                }
            )

    population.sort(
        key=lambda row: (
            str(row["scene_id"]),
            str(row["episode_id"]),
            int(row["decision_step"]),
            int(row["feature_row_index"]),
        )
    )
    event_ids = [str(row["event_id"]) for row in population]
    if len(event_ids) != len(set(event_ids)):
        raise ProtocolError("duplicate MF3ZU event identity")
    event_order = {event_id: index for index, event_id in enumerate(event_ids)}
    exact_targets.sort(key=lambda row: event_order[str(row["event_id"])])
    if [str(row["event_id"]) for row in exact_targets] != event_ids:
        raise ProtocolError("sanitized population/exact-target identity drift")

    population_episodes = {str(row["episode_id"]) for row in population}
    population_scenes = {str(row["scene_id"]) for row in population}

    summary: dict[str, object] = {
        "source_rows": source_row_count,
        "candidate_eligible_rows": candidate_eligible_count,
        "population_rows": len(population),
        "exact_target_rows": len(exact_targets),
        "feature_row_physical_step_mismatch_rows": selected_feature_physical_mismatch_count,
        "source_episodes": len(source_episodes),
        "source_raw_scenes": len(set(scenes)),
        "episodes": len(population_episodes),
        "raw_scenes": len(population_scenes),
        "scene_fold_mapping": dict(sorted(folds.items())),
        "fold_scene_counts": {
            str(fold): sum(value == fold for value in folds.values())
            for fold in range(FOLDS)
        },
        "fold_decision_counts": {
            str(fold): sum(int(row["scene_fold"]) == fold for row in population)
            for fold in range(FOLDS)
        },
        "exact_target_accessed_for_support_eligibility": True,
        "target_value_in_sanitized_population": False,
        "baseline_score_or_correctness_accessed": False,
        "outcome_or_utility_accessed": False,
    }
    if enforce_frozen_counts:
        expected = {
            "source_rows": EXPECTED_SOURCE_ROWS,
            "candidate_eligible_rows": EXPECTED_CANDIDATE_ELIGIBLE_ROWS,
            "population_rows": EXPECTED_POPULATION_ROWS,
            "exact_target_rows": EXPECTED_POPULATION_ROWS,
            "source_episodes": EXPECTED_SOURCE_EPISODES,
            "source_raw_scenes": EXPECTED_SOURCE_SCENES,
            "episodes": EXPECTED_POPULATION_EPISODES,
            "raw_scenes": EXPECTED_POPULATION_SCENES,
        }
        actual = {name: summary[name] for name in expected}
        if actual != expected:
            raise ProtocolError(f"MF3ZU frozen population count drift: {actual}")
    return population, exact_targets, summary


def _canonical_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError(f"stale immutable-artifact partial: {partial}")
    partial.write_bytes(payload)
    os.replace(partial, path)


def write_population() -> dict[str, object]:
    """Materialize immutable sanitized-population and exact-target artifacts."""

    verify_protocol()
    if any(
        path.exists() or path.is_symlink()
        for path in (
            POPULATION_PATH,
            POPULATION_MANIFEST_PATH,
            EXACT_TARGETS_PATH,
            EVIDENCE_MEMORY_PATH,
            EVIDENCE_MEMORY_MANIFEST_PATH,
            RESULT_PATH,
        )
    ):
        raise ProtocolError("MF3ZU population or downstream artifact already exists")

    rows, exact_targets, summary = build_population_rows()
    payload = _canonical_jsonl(rows)
    target_payload = _canonical_jsonl(exact_targets)
    population_sha = hashlib.sha256(payload).hexdigest()
    exact_targets_sha = hashlib.sha256(target_payload).hexdigest()
    manifest = {
        "schema_version": "revealnav-mf3zu-rxr-feasibility-population-manifest/1",
        "revision": REVISION,
        "status": "MF3ZU_RXR_EXACT_SUPPORT_POPULATION_FROZEN",
        "population": {
            "path": str(POPULATION_PATH.relative_to(ROOT)),
            "bytes": len(payload),
            "sha256": population_sha,
            "contains_target_fields": False,
            "authorized_for_annotation_and_memory_required": True,
        },
        "exact_targets": {
            "path": str(EXACT_TARGETS_PATH.relative_to(ROOT)),
            "bytes": len(target_payload),
            "sha256": exact_targets_sha,
            "rows": len(exact_targets),
            "authorized_for_annotation_or_memory_required": False,
            "trainer_access_requires_evidence_manifest_frozen": True,
        },
        **summary,
        "selection": {
            "rule": "include every MF3B row with candidate_mask count >= 2 and exact target feature slot active",
            "sampling": False,
            "allowed_inputs": [
                "dataset_scene_episode_identity",
                "feature_row_identity",
                "candidate_mask",
                "target_index_for_exact_support_existence_only",
                "causal_replay_provenance",
            ],
            "forbidden_inputs": [
                "frozen_ETP_target_correctness",
                "native_score_rank_of_target",
                "reward",
                "utility",
                "success",
                "route_outcome",
                "future_observation",
            ],
        },
        "target_access_boundary": {
            "support_existence_checked_before_evidence": True,
            "target_values_in_sanitized_population": False,
            "annotation_must_not_open_exact_targets": True,
            "memory_required_must_not_open_exact_targets": True,
            "training_value_access": "ONLY_AFTER_EVIDENCE_MEMORY_MANIFEST_FROZEN",
        },
        "evidence_memory_frozen": False,
        "public_split_access": dict(PUBLIC_CLOSED),
        "full_navigation_run": False,
        "checkpoint_generated": False,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    # Write the data first.  A missing manifest is an explicit incomplete state,
    # never a silently accepted population.
    _atomic_write_once(POPULATION_PATH, payload)
    try:
        _atomic_write_once(EXACT_TARGETS_PATH, target_payload)
        _atomic_write_once(POPULATION_MANIFEST_PATH, manifest_payload)
    except Exception:
        # Deliberately retain the population rather than destructively deleting
        # an immutable artifact; the next invocation fails closed for review.
        raise
    return manifest


def _source_inventory() -> dict[str, dict[str, object]]:
    return {
        name: inventory(path, expected_sha256=EXPECTED_SOURCE_SHA256[name])
        for name, path in SOURCE_PATHS.items()
    }


def _implementation_files() -> tuple[str, ...]:
    tests = tuple(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "tests").glob("test_mf3zu_*.py"))
    )
    if not tests:
        raise ProtocolError("MF3ZU implementation has no regression tests")
    return IMPLEMENTATION_FILES + tests


def build_protocol() -> dict[str, object]:
    _ensure_base_commit()
    source_inventory = _source_inventory()
    implementation_inventory = {
        name: inventory(ROOT / name) for name in _implementation_files()
    }
    mf3b_protocol = _read_object(MF3B_TARGET_PROTOCOL)
    if mf3b_protocol.get("dataset") != "RxR train guide en-US/en-IN only":
        raise ProtocolError("MF3B RxR train scope drift")
    if mf3b_protocol.get("public_unseen_authorized") is not False:
        raise ProtocolError("MF3B source opens public unseen")
    mf3zt_result = _read_object(MF3ZT_RESULT)
    if mf3zt_result.get("final_pass_fail") != "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL":
        raise ProtocolError("immutable MF3ZT failure status drift")

    return {
        "schema_version": "revealnav-mf3zu-rxr-evidence-memory-feasibility-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_MF3ZU_FEASIBILITY_POPULATION_AND_RESULTS",
        "source_commit": BASE_COMMIT,
        "seal_commit": current_commit(),
        "scientific_question": "Does explicit instruction-conditioned semantic evidence memory improve held-scene frozen-ETP RxR candidate decisions?",
        "revision_relationship": {
            "new_RxR_only_revision": True,
            "R2R_in_scope": False,
            "MF3ZT_modified": False,
            "MF3ZT_failure_preserved": True,
            "may_be_reported_as_MF3ZT_two_domain_pass": False,
        },
        "scope": {
            "dataset": DATASET,
            "train_development_only": True,
            "candidate_ranking_feasibility_only": True,
            "train_native_observation_replay_authorized": True,
            "observation_replay_purpose": "causal prefix evidence materialization for the fixed population only",
            "full_navigation": False,
            "SR_SPL_optimization": False,
            "UAD_reveal_expiry_oracle_returnability": False,
            "policy_gradient_or_RL": False,
            "ETP_fine_tuning": False,
        },
        "source_inventory": source_inventory,
        "implementation_inventory": implementation_inventory,
        "population": {
            "source": "MF3B RxR train-guide frozen ETP online features",
            "source_rows": EXPECTED_SOURCE_ROWS,
            "source_episodes": EXPECTED_SOURCE_EPISODES,
            "source_raw_scenes": EXPECTED_SOURCE_SCENES,
            "candidate_eligible_rows_before_exact_support": EXPECTED_CANDIDATE_ELIGIBLE_ROWS,
            "selection_rule": "candidate_mask_count>=2_and_exact_target_feature_slot_active",
            "sampling": False,
            "expected_rows": EXPECTED_POPULATION_ROWS,
            "expected_episodes": EXPECTED_POPULATION_EPISODES,
            "expected_raw_scenes": EXPECTED_POPULATION_SCENES,
            "exact_target_accessed_for_support_eligibility": True,
            "target_value_in_sanitized_population": False,
            "baseline_score_or_correctness_selection": False,
            "outcome_or_utility_selection": False,
            "artifact_status": "NOT_YET_MATERIALIZED",
        },
        "exact_target_boundary": {
            "source": "preexisting exact MF3B RxR current-candidate teacher",
            "exact_same_episode_prefix": True,
            "candidate_aligned": True,
            "support_existence_used_to_fix_population": True,
            "sanitized_population_and_exact_targets_are_separate_artifacts": True,
            "exact_target_rows": EXPECTED_POPULATION_ROWS,
            "annotation_may_inventory_exact_targets": False,
            "annotation_may_open_exact_targets": False,
            "memory_required_may_inventory_exact_targets": False,
            "memory_required_may_open_exact_targets": False,
            "trainer_may_open_exact_targets_before_evidence_manifest_frozen": False,
            "target_coordinate_system": "MF3B_candidate_feature_slot",
        },
        "feature_to_physical_step_mapping": {
            "feature_row_equals_physical_step_assumed": False,
            "pairing": "ascending_candidate_eligible_feature_ordinal_to_hash_chained_shadow_decision_ordinal",
            "required_checks": [
                "eligible_feature_count_equals_shadow_decision_count",
                "shadow_previous_hash_chain",
                "shadow_physical_steps_strictly_increase",
                "each_shadow_physical_step_resolves_to_exactly_one_native_trace_row",
                "candidate_cardinality_matches_without_coordinate_binding",
            ],
            "feature_slots_graph_indices_action_ids_bound_before_replay": False,
            "binding_requires": "replay_embedding_and_score_byte_match",
        },
        "scene_split": {
            "folds": FOLDS,
            "cluster": "raw_MP3D_scene",
            "algorithm": "salted_SHA256_scene_order_then_round_robin",
            "salt": FOLD_SALT,
            "old_MF3B_split_ignored": True,
        },
        "evidence": {
            "ontology": list(EVIDENCE_ONTOLOGY),
            "confidence_classes": list(CONFIDENCE_CLASSES),
            "K_MEM": K_MEM,
            "retrieval_trainable": False,
            "retrieval_order": [
                "active_instruction_atom_order",
                "source_step_descending",
                "evidence_id",
            ],
            "mean_pooling": True,
            "whole_history_RGB_as_model_input": False,
            "generic_history_embedding_as_memory_input": False,
            "future_evidence_forbidden": True,
            "materialized_and_hashed_before_training_target_value_access": True,
            "fixed_record_feature": {
                "record_dimensions": EVIDENCE_RECORD_DIM,
                "ontology_one_hot": 5,
                "historical_confidence_one_hot": 3,
                "current_status_one_hot": 3,
                "log1p_age": 1,
                "reciprocal_recency": 1,
                "signed_SHA_token_hash": 64,
                "candidate_binding": CANDIDATE_BINDING_DIM,
                "candidate_feature_dimensions": EVIDENCE_FEATURE_DIM,
                "pooling": "mean_per_candidate_over_at_most_8_records",
            },
            "extractor": dict(QWEN_EXTRACTOR),
            "human_review": {
                "status": "SKIPPED_BY_USER_FOR_THIS_ATTEMPT",
                "human_verified": False,
                "gold_labels": False,
                "may_be_described_as_human_validated": False,
            },
        },
        "memory_required_definition": {
            "MEMORY_REQUIRED": "instruction-relevant evidence appeared in the causal past, is absent or insufficient now, and is semantically needed for the current candidate decision",
            "MEMORY_NOT_REQUIRED": "current observation is sufficient or the decision does not require historical semantic state",
            "classified_without_opening_exact_target_artifact": True,
            "allowed_inputs": [
                "instruction_semantics",
                "causal_visual_history",
                "current_visual_observation",
                "current_candidate_geometry_or_appearance",
                "instruction_constraint_structure",
            ],
            "forbidden_inputs": [
                "candidate_target",
                "success",
                "reward",
                "utility",
                "future_frame",
            ],
        },
        "frozen_ETP": {
            "ETP_frozen": True,
            "candidate_generator_frozen": True,
            "visual_backbone_frozen": True,
            "topology_encoder_frozen": True,
            "checkpoint": source_inventory["rxr_etp_checkpoint"],
            "shared_backbone": source_inventory["etp_backbone"],
        },
        "model_and_training": {
            "arms": list(ARMS),
            "A": "original frozen ETP masked native_scores; no training",
            "B_and_C_common_architecture": {
                "candidate_projection": "Linear(768,64)",
                "evidence_projection": f"Linear({EVIDENCE_FEATURE_DIM},64)",
                "interaction": "concat(h,m,h_elementwise_m)",
                "residual_head": "Linear(192,64)-GELU-Linear(64,1)",
                "output": "frozen_ETP_base_score_plus_residual",
            },
            "evidence_pooling": "mean per candidate",
            "loss": "candidate_set_cross_entropy",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 40,
            "seed": TRAINING_SEED,
            "early_stopping": False,
            "best_checkpoint_selection": False,
            "B_C_common_initialization": True,
            "B_C_common_batch_order": True,
            "architecture_sweep": False,
            "threshold_search": False,
            "hyperparameter_grid": False,
            "multi_seed_rescue": False,
            "shuffled_memory": {
                "different_event_within_training_fold": True,
                "preserve_count_and_feature_distribution": True,
                "held_fold_target_or_memory_as_donor": False,
            },
        },
        "evaluation": {
            "folds": FOLDS,
            "split_unit": "raw_MP3D_scene",
            "OOF_complete_required": True,
            "standardization_fit_on_train_fold_only": True,
            "metrics": ["Acc@1", "MRR", "MeanRank", "pairwise_accuracy"],
            "subgroups": ["ALL", "MEMORY_REQUIRED", "MEMORY_NOT_REQUIRED"],
            "bootstrap": {
                "cluster": "raw_MP3D_scene",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "pass_fail": {
            "domain": DATASET,
            "R2R_required": False,
            "memory_required_min_decisions": MIN_MEMORY_REQUIRED_DECISIONS,
            "memory_required_min_raw_scenes": MIN_MEMORY_REQUIRED_SCENES,
            "memory_required_B_minus_A_Acc_positive": True,
            "memory_required_B_minus_A_Acc_lower95_positive": True,
            "memory_required_B_minus_A_MRR_positive": True,
            "memory_required_B_minus_C_Acc_positive": True,
            "memory_required_B_minus_C_Acc_lower95_positive": True,
            "memory_not_required_B_minus_A_Acc_min": -0.01,
            "all_B_minus_A_Acc_min": 0.0,
            "status_on_support_fail": "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
            "status_on_specificity_fail": "MF3ZU_RXR_EVIDENCE_SPECIFICITY_FAIL",
            "status_on_pass": "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS",
            "status_on_fail": "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL",
        },
        "execution": {
            "population_built": False,
            "observation_replay_run": False,
            "evidence_memory_built": False,
            "exact_target_values_opened_for_training": False,
            "training_started": False,
            "OOF_evaluation_run": False,
            "bootstrap_run": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
            "checkpoint_for_deployment": False,
            "public_split_access": dict(PUBLIC_CLOSED),
        },
        "public_split_access": dict(PUBLIC_CLOSED),
    }


def seal_protocol() -> dict[str, object]:
    if PROTOCOL_PATH.exists() or PROTOCOL_PATH.is_symlink():
        raise ProtocolError("MF3ZU protocol already sealed; refusing overwrite")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            POPULATION_PATH,
            POPULATION_MANIFEST_PATH,
            EXACT_TARGETS_PATH,
            EVIDENCE_MEMORY_PATH,
            EVIDENCE_MEMORY_MANIFEST_PATH,
            RESULT_PATH,
        )
    ):
        raise ProtocolError("MF3ZU downstream material exists before protocol seal")
    value = build_protocol()
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_once(PROTOCOL_PATH, payload)
    return value


def verify_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    _ensure_base_commit()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("MF3ZU protocol is missing")
    value = _read_object(path)
    if value.get("revision") != REVISION:
        raise ProtocolError("MF3ZU revision drift")
    if value.get("status") != "SEALED_BEFORE_MF3ZU_FEASIBILITY_POPULATION_AND_RESULTS":
        raise ProtocolError("MF3ZU protocol status drift")
    if value.get("source_commit") != BASE_COMMIT:
        raise ProtocolError("MF3ZU source commit drift")
    if value.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZU public split opened")
    relationship = value.get("revision_relationship")
    if not isinstance(relationship, Mapping):
        raise ProtocolError("MF3ZU revision relationship missing")
    if relationship.get("R2R_in_scope") is not False:
        raise ProtocolError("MF3ZU unexpectedly includes R2R")
    if relationship.get("MF3ZT_modified") is not False:
        raise ProtocolError("MF3ZU claims an MF3ZT mutation")
    if relationship.get("MF3ZT_failure_preserved") is not True:
        raise ProtocolError("MF3ZU does not preserve the MF3ZT failure")
    scope = value.get("scope")
    if not isinstance(scope, Mapping):
        raise ProtocolError("MF3ZU scope section missing")
    if scope.get("dataset") != DATASET:
        raise ProtocolError("MF3ZU dataset scope drift")
    if scope.get("train_development_only") is not True:
        raise ProtocolError("MF3ZU train-development boundary drift")
    if scope.get("candidate_ranking_feasibility_only") is not True:
        raise ProtocolError("MF3ZU feasibility-only boundary drift")
    if scope.get("full_navigation") is not False:
        raise ProtocolError("MF3ZU scope opens full navigation")

    population = value.get("population")
    if not isinstance(population, Mapping):
        raise ProtocolError("MF3ZU population section missing")
    expected_population = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "source_episodes": EXPECTED_SOURCE_EPISODES,
        "source_raw_scenes": EXPECTED_SOURCE_SCENES,
        "candidate_eligible_rows_before_exact_support": EXPECTED_CANDIDATE_ELIGIBLE_ROWS,
        "selection_rule": "candidate_mask_count>=2_and_exact_target_feature_slot_active",
        "sampling": False,
        "expected_rows": EXPECTED_POPULATION_ROWS,
        "expected_episodes": EXPECTED_POPULATION_EPISODES,
        "expected_raw_scenes": EXPECTED_POPULATION_SCENES,
        "exact_target_accessed_for_support_eligibility": True,
        "target_value_in_sanitized_population": False,
        "baseline_score_or_correctness_selection": False,
        "outcome_or_utility_selection": False,
        "artifact_status": "NOT_YET_MATERIALIZED",
    }
    for key, expected in expected_population.items():
        if population.get(key) != expected:
            raise ProtocolError(f"MF3ZU population {key} drift")

    target_boundary = value.get("exact_target_boundary")
    expected_target_boundary = {
        "source": "preexisting exact MF3B RxR current-candidate teacher",
        "exact_same_episode_prefix": True,
        "candidate_aligned": True,
        "support_existence_used_to_fix_population": True,
        "sanitized_population_and_exact_targets_are_separate_artifacts": True,
        "exact_target_rows": EXPECTED_POPULATION_ROWS,
        "annotation_may_inventory_exact_targets": False,
        "annotation_may_open_exact_targets": False,
        "memory_required_may_inventory_exact_targets": False,
        "memory_required_may_open_exact_targets": False,
        "trainer_may_open_exact_targets_before_evidence_manifest_frozen": False,
        "target_coordinate_system": "MF3B_candidate_feature_slot",
    }
    if not isinstance(target_boundary, Mapping) or dict(target_boundary) != expected_target_boundary:
        raise ProtocolError("MF3ZU exact-target access boundary drift")

    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ProtocolError("MF3ZU evidence section missing")
    if evidence.get("ontology") != list(EVIDENCE_ONTOLOGY):
        raise ProtocolError("MF3ZU evidence ontology drift")
    if evidence.get("K_MEM") != K_MEM:
        raise ProtocolError("MF3ZU retrieval budget drift")
    if evidence.get("extractor") != QWEN_EXTRACTOR:
        raise ProtocolError("MF3ZU extractor drift")
    expected_record_feature = {
        "record_dimensions": EVIDENCE_RECORD_DIM,
        "ontology_one_hot": 5,
        "historical_confidence_one_hot": 3,
        "current_status_one_hot": 3,
        "log1p_age": 1,
        "reciprocal_recency": 1,
        "signed_SHA_token_hash": 64,
        "candidate_binding": CANDIDATE_BINDING_DIM,
        "candidate_feature_dimensions": EVIDENCE_FEATURE_DIM,
        "pooling": "mean_per_candidate_over_at_most_8_records",
    }
    if evidence.get("fixed_record_feature") != expected_record_feature:
        raise ProtocolError("MF3ZU fixed evidence feature drift")
    expected_human_review = {
        "status": "SKIPPED_BY_USER_FOR_THIS_ATTEMPT",
        "human_verified": False,
        "gold_labels": False,
        "may_be_described_as_human_validated": False,
    }
    if evidence.get("human_review") != expected_human_review:
        raise ProtocolError("MF3ZU human-review boundary drift")

    memory_required = value.get("memory_required_definition")
    if not isinstance(memory_required, Mapping):
        raise ProtocolError("MF3ZU memory-required definition missing")
    if memory_required.get("classified_without_opening_exact_target_artifact") is not True:
        raise ProtocolError("MF3ZU memory-required target blindness drift")
    forbidden_inputs = memory_required.get("forbidden_inputs")
    if not isinstance(forbidden_inputs, list) or "candidate_target" not in forbidden_inputs:
        raise ProtocolError("MF3ZU memory-required forbidden inputs drift")

    model = value.get("model_and_training")
    expected_model = {
        "arms": list(ARMS),
        "A": "original frozen ETP masked native_scores; no training",
        "B_and_C_common_architecture": {
            "candidate_projection": "Linear(768,64)",
            "evidence_projection": f"Linear({EVIDENCE_FEATURE_DIM},64)",
            "interaction": "concat(h,m,h_elementwise_m)",
            "residual_head": "Linear(192,64)-GELU-Linear(64,1)",
            "output": "frozen_ETP_base_score_plus_residual",
        },
        "evidence_pooling": "mean per candidate",
        "loss": "candidate_set_cross_entropy",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "epochs": 40,
        "seed": TRAINING_SEED,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "B_C_common_initialization": True,
        "B_C_common_batch_order": True,
        "architecture_sweep": False,
        "threshold_search": False,
        "hyperparameter_grid": False,
        "multi_seed_rescue": False,
        "shuffled_memory": {
            "different_event_within_training_fold": True,
            "preserve_count_and_feature_distribution": True,
            "held_fold_target_or_memory_as_donor": False,
        },
    }
    if not isinstance(model, Mapping) or dict(model) != expected_model:
        raise ProtocolError("MF3ZU fixed model/training configuration drift")

    split = value.get("scene_split")
    if not isinstance(split, Mapping) or split.get("salt") != FOLD_SALT:
        raise ProtocolError("MF3ZU scene-fold protocol drift")

    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ProtocolError("MF3ZU evaluation section missing")
    expected_bootstrap = {
        "cluster": "raw_MP3D_scene",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    if evaluation.get("bootstrap") != expected_bootstrap:
        raise ProtocolError("MF3ZU bootstrap configuration drift")

    expected_pass_fail = {
        "domain": DATASET,
        "R2R_required": False,
        "memory_required_min_decisions": MIN_MEMORY_REQUIRED_DECISIONS,
        "memory_required_min_raw_scenes": MIN_MEMORY_REQUIRED_SCENES,
        "memory_required_B_minus_A_Acc_positive": True,
        "memory_required_B_minus_A_Acc_lower95_positive": True,
        "memory_required_B_minus_A_MRR_positive": True,
        "memory_required_B_minus_C_Acc_positive": True,
        "memory_required_B_minus_C_Acc_lower95_positive": True,
        "memory_not_required_B_minus_A_Acc_min": -0.01,
        "all_B_minus_A_Acc_min": 0.0,
        "status_on_support_fail": "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
        "status_on_specificity_fail": "MF3ZU_RXR_EVIDENCE_SPECIFICITY_FAIL",
        "status_on_pass": "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS",
        "status_on_fail": "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL",
    }
    pass_fail = value.get("pass_fail")
    if not isinstance(pass_fail, Mapping) or dict(pass_fail) != expected_pass_fail:
        raise ProtocolError("MF3ZU PASS/FAIL gates drift")

    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise ProtocolError("MF3ZU execution section missing")
    for key in (
        "population_built",
        "observation_replay_run",
        "evidence_memory_built",
        "exact_target_values_opened_for_training",
        "training_started",
        "OOF_evaluation_run",
        "bootstrap_run",
        "full_navigation_run",
        "checkpoint_generated",
        "checkpoint_for_deployment",
    ):
        if execution.get(key) is not False:
            raise ProtocolError(f"MF3ZU protocol prematurely marks {key}")
    if execution.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZU execution public split opened")

    for section in ("source_inventory", "implementation_inventory"):
        records = value.get(section)
        if not isinstance(records, Mapping):
            raise ProtocolError(f"malformed MF3ZU {section}")
        for name, item in records.items():
            if not isinstance(item, Mapping):
                raise ProtocolError(f"malformed MF3ZU inventory item: {name}")
            current = inventory(ROOT / str(item["path"]))
            if current != dict(item):
                raise ProtocolError(f"MF3ZU inventory drift: {item['path']}")
    implementation_records = value.get("implementation_inventory")
    if not isinstance(implementation_records, Mapping):
        raise ProtocolError("MF3ZU implementation inventory is malformed")
    if set(implementation_records) != set(_implementation_files()):
        raise ProtocolError("MF3ZU implementation or regression-test set drift")
    return value


__all__ = [
    "ROOT",
    "REVISION",
    "OUTPUT",
    "PROTOCOL_PATH",
    "POPULATION_PATH",
    "POPULATION_MANIFEST_PATH",
    "EXACT_TARGETS_PATH",
    "EVIDENCE_MEMORY_PATH",
    "EVIDENCE_MEMORY_MANIFEST_PATH",
    "RESULT_PATH",
    "BASE_COMMIT",
    "DATASET",
    "PUBLIC_CLOSED",
    "FOLDS",
    "FOLD_SALT",
    "MIN_CANDIDATES",
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_CANDIDATE_ELIGIBLE_ROWS",
    "EXPECTED_POPULATION_ROWS",
    "EXPECTED_SOURCE_EPISODES",
    "EXPECTED_SOURCE_SCENES",
    "EXPECTED_POPULATION_EPISODES",
    "EXPECTED_POPULATION_SCENES",
    "EVIDENCE_ONTOLOGY",
    "CONFIDENCE_CLASSES",
    "ARMS",
    "K_MEM",
    "EVIDENCE_RECORD_DIM",
    "CANDIDATE_BINDING_DIM",
    "EVIDENCE_FEATURE_DIM",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "TRAINING_SEED",
    "MIN_MEMORY_REQUIRED_DECISIONS",
    "MIN_MEMORY_REQUIRED_SCENES",
    "QWEN_EXTRACTOR",
    "MF3B_TARGET_PROTOCOL",
    "MF3B_TARGET_MANIFEST",
    "RXR_TRAIN_GUIDE",
    "RXR_ETP_CHECKPOINT",
    "ETP_BACKBONE",
    "EXPECTED_SOURCE_SHA256",
    "SOURCE_PATHS",
    "IMPLEMENTATION_FILES",
    "FORBIDDEN_POPULATION_FIELDS",
    "ProtocolError",
    "sha256_file",
    "inventory",
    "scene_fold_mapping",
    "build_population_rows",
    "write_population",
    "build_protocol",
    "seal_protocol",
    "verify_protocol",
]
