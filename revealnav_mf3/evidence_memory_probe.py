"""Fail-closed candidate-target support audit for MF3ZT.

The scientific probe is conditional on legal candidate-ranking targets in both
domains.  The current revision deliberately implements only this prerequisite.
If it fails, no decision population, evidence memory, learner, OOF prediction,
or bootstrap object may be produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .mf3zt_protocol import (
    DOMAINS,
    ETP_TRAINER,
    MF3ZK_R2R_OUTCOME_MANIFEST,
    MF3ZP_OBSERVATION_STATUS,
    MF3ZQ_POPULATION,
    MF3ZR_BINDING_AUDIT,
    OUTPUT,
    PROTOCOL_PATH,
    PUBLIC_CLOSED,
    RESULT_PATH,
    REVISION,
    ROOT,
    R2R_TRAIN_GT,
    RXR_TARGET_MANIFEST,
    RXR_TARGET_PROTOCOL,
    TARGET_AUDIT_PATH,
    inventory,
    sha256_file,
    verify_protocol,
)


TARGET_FIELD_NAMES = frozenset(
    {
        "target",
        "target_index",
        "target_candidate_id",
        "target_candidate_ids",
        "teacher_action",
        "teacher_action_id",
        "teacher_index",
        "correct_action",
        "correct_action_id",
        "correct_candidate",
        "correct_candidate_id",
        "correct_candidate_ids",
        "expert_action",
        "expert_action_id",
        "positive_candidate_ids",
        "teacher_target",
    }
)
ALLOWED_TARGET_PROVENANCE = frozenset(
    {
        "exact_train_native_action_or_candidate_supervision",
        "exact_same_episode_same_prefix_branch_supervision",
        "existing_frozen_causal_decision_target",
    }
)
FORBIDDEN_TARGET_PROVENANCE = frozenset(
    {
        "frozen_native_action_self_label",
        "route_truth_reconstruction",
        "shortest_path_reconstruction",
        "nearest_candidate_mapping",
        "route_level_outcome",
        "utility",
        "public_split",
        "cross_episode_pairing",
        "post_hoc_label",
    }
)
DOWNSTREAM_FILENAMES = (
    "MF3ZT_DECISION_POPULATION.jsonl",
    "MF3ZT_EVIDENCE_MEMORY.jsonl",
    "MF3ZT_FOLD_ASSIGNMENTS.json",
    "MF3ZT_OOF_PREDICTIONS.jsonl",
    "MF3ZT_BOOTSTRAP.json",
    "MF3ZT_RERANKER.pt",
    "MF3ZT_RERANKER.pth",
    "MF3ZT_RERANKER.ckpt",
)


class ProbeAuditError(RuntimeError):
    """Raised on malformed, drifting, or outcome-contaminated audit input."""


@dataclass(frozen=True)
class TargetSourceSummary:
    """A normalized candidate-target source used by the support gate."""

    source_id: str
    dataset: str
    provenance: str
    preexisting: bool
    train_development_only: bool
    public_split_accessed: bool
    exact_same_episode_prefix: bool
    exact_candidate_set_alignment: bool
    target_rows: int
    rankable_target_rows: int
    raw_scene_count: int
    unique_episode_count: int
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.dataset not in DOMAINS:
            raise ProbeAuditError(f"unknown target-source domain: {self.dataset}")
        if self.target_rows < 0 or self.rankable_target_rows < 0:
            raise ProbeAuditError("negative target-source count")
        if self.rankable_target_rows > self.target_rows:
            raise ProbeAuditError("rankable target count exceeds target count")
        if self.raw_scene_count < 0 or self.unique_episode_count < 0:
            raise ProbeAuditError("negative target-source support count")
        structurally_legal = (
            self.preexisting
            and self.train_development_only
            and not self.public_split_accessed
            and self.exact_same_episode_prefix
            and self.exact_candidate_set_alignment
            and self.rankable_target_rows > 0
            and self.provenance in ALLOWED_TARGET_PROVENANCE
            and self.provenance not in FORBIDDEN_TARGET_PROVENANCE
            and not self.rejection_reasons
        )
        if self.accepted != structurally_legal:
            raise ProbeAuditError(f"accepted flag disagrees with source contract: {self.source_id}")


def evaluate_target_support(
    sources: Sequence[TargetSourceSummary],
) -> dict[str, object]:
    """Apply the fixed both-domain target-support prerequisite."""

    accepted_by_domain: dict[str, list[TargetSourceSummary]] = {domain: [] for domain in DOMAINS}
    rejected_by_domain: dict[str, list[TargetSourceSummary]] = {domain: [] for domain in DOMAINS}
    for source in sources:
        source.validate()
        (accepted_by_domain if source.accepted else rejected_by_domain)[source.dataset].append(source)

    domain_support: dict[str, object] = {}
    for domain in DOMAINS:
        accepted = accepted_by_domain[domain]
        rejected = rejected_by_domain[domain]
        target_rows = sum(source.target_rows for source in accepted)
        rankable = sum(source.rankable_target_rows for source in accepted)
        domain_support[domain] = {
            "status": "SUPPORTED" if rankable > 0 else "UNSUPPORTED",
            "accepted_source_ids": [source.source_id for source in accepted],
            "rejected_source_ids": [source.source_id for source in rejected],
            "legal_target_rows": target_rows,
            "legal_rankable_target_rows": rankable,
            "legal_raw_scene_count_lower_bound": max(
                (source.raw_scene_count for source in accepted), default=0
            ),
            "legal_unique_episode_count_lower_bound": max(
                (source.unique_episode_count for source in accepted), default=0
            ),
            "accepted_source_count": len(accepted),
        }
    passed = all(domain_support[domain]["status"] == "SUPPORTED" for domain in DOMAINS)
    return {
        "passed": passed,
        "status": "MF3ZT_DECISION_TARGET_SUPPORT_PASS" if passed else "MF3ZT_DECISION_TARGET_SUPPORT_FAIL",
        "required_domains": list(DOMAINS),
        "domain_support": domain_support,
        "first_failure": None if passed else "decision_target_support",
    }


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeAuditError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProbeAuditError(f"expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


def _project_path(raw: object) -> Path:
    path = ROOT / str(raw)
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents or path.is_symlink():
        raise ProbeAuditError(f"non-project-local source path: {raw}")
    return path


def _check_inventory_record(raw: Mapping[str, object]) -> Path:
    path = _project_path(raw.get("path"))
    if not path.is_file():
        raise ProbeAuditError(f"missing source file: {path}")
    if int(raw.get("bytes", -1)) != path.stat().st_size:
        raise ProbeAuditError(f"source byte-count drift: {path}")
    if raw.get("sha256") != sha256_file(path):
        raise ProbeAuditError(f"source hash drift: {path}")
    return path


def _numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised by the audit runtime
        raise ProbeAuditError(
            "MF3ZT source-array audit requires the project-local etpr1 environment"
        ) from error
    return np


def _audit_rxr_exact_targets() -> tuple[TargetSourceSummary, dict[str, object]]:
    np = _numpy()
    protocol = _read_object(RXR_TARGET_PROTOCOL)
    manifest = _read_object(RXR_TARGET_MANIFEST)
    if protocol.get("dataset") != "RxR train guide en-US/en-IN only":
        raise ProbeAuditError("MF3B target source is not RxR train")
    if protocol.get("public_unseen_authorized") is not False:
        raise ProbeAuditError("MF3B target protocol opened public unseen")
    labels = protocol.get("labels")
    if not isinstance(labels, Mapping) or labels.get("target") != "native RxR nDTW teacher current ghost":
        raise ProbeAuditError("MF3B exact-target provenance drift")
    if manifest.get("status") != "PASS" or manifest.get("failures") != []:
        raise ProbeAuditError("MF3B target manifest is not complete")
    if manifest.get("public_unseen_authorized") is not False:
        raise ProbeAuditError("MF3B target manifest opened public unseen")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ProbeAuditError("MF3B target manifest has no records")

    rows = 0
    target_rows = 0
    rankable_target_rows = 0
    scenes: set[str] = set()
    episodes: set[str] = set()
    observed_array_keys: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ProbeAuditError("malformed MF3B manifest record")
        if record.get("future_teacher_used_as_online_input") is not False:
            raise ProbeAuditError("MF3B future teacher entered online input")
        path = _check_inventory_record(record)
        scenes.add(str(record.get("scene_id")))
        episodes.add(str(record.get("episode_id")))
        with np.load(path, allow_pickle=False) as arrays:
            observed_array_keys.update(arrays.files)
            required = {"candidate_mask", "target_index", "target_in_set"}
            if not required.issubset(arrays.files):
                raise ProbeAuditError(f"MF3B target arrays missing at {path}")
            mask = arrays["candidate_mask"]
            target_index = arrays["target_index"]
            target_in_set = arrays["target_in_set"]
            if mask.ndim != 2 or target_index.ndim != 1 or target_in_set.ndim != 1:
                raise ProbeAuditError(f"MF3B target-array rank drift at {path}")
            if mask.shape[0] != target_index.shape[0] or mask.shape[0] != target_in_set.shape[0]:
                raise ProbeAuditError(f"MF3B target-array length drift at {path}")
            if int(record.get("steps", -1)) != mask.shape[0]:
                raise ProbeAuditError(f"MF3B manifest step-count drift at {path}")
            for index in range(mask.shape[0]):
                rows += 1
                target = int(target_index[index])
                present = target >= 0
                labeled_present = float(target_in_set[index]) >= 0.5
                if present != labeled_present:
                    raise ProbeAuditError(f"MF3B target-in-set mismatch at {path}:{index}")
                if not present:
                    continue
                if target >= mask.shape[1] or not bool(mask[index, target]):
                    raise ProbeAuditError(f"MF3B target is outside current candidate set at {path}:{index}")
                target_rows += 1
                if int(mask[index].sum()) >= 2:
                    rankable_target_rows += 1

    summary = TargetSourceSummary(
        source_id="mf3b_rxr_exact_current_ghost_teacher",
        dataset="RxR",
        provenance="exact_train_native_action_or_candidate_supervision",
        preexisting=True,
        train_development_only=True,
        public_split_accessed=False,
        exact_same_episode_prefix=True,
        exact_candidate_set_alignment=True,
        target_rows=target_rows,
        rankable_target_rows=rankable_target_rows,
        raw_scene_count=len(scenes),
        unique_episode_count=len(episodes),
        accepted=True,
    )
    summary.validate()
    return summary, {
        "source_id": summary.source_id,
        "source_protocol": inventory(RXR_TARGET_PROTOCOL),
        "source_manifest": inventory(RXR_TARGET_MANIFEST),
        "manifest_records": len(records),
        "decision_rows": rows,
        "target_rows": target_rows,
        "rankable_target_rows": rankable_target_rows,
        "raw_scene_count": len(scenes),
        "unique_episode_count": len(episodes),
        "observed_array_keys": sorted(observed_array_keys),
        "teacher_role": "label_only_after_native_policy_output",
        "future_teacher_used_as_online_input": False,
        "accepted": True,
    }


def _target_like_fields(fields: Iterable[str]) -> set[str]:
    lowered = {str(field).lower() for field in fields}
    return lowered & TARGET_FIELD_NAMES


def _audit_frozen_observation_corpus() -> tuple[list[TargetSourceSummary], dict[str, object]]:
    np = _numpy()
    population = _read_jsonl(MF3ZQ_POPULATION)
    if len(population) != 80:
        raise ProbeAuditError("fixed MF3ZQ observation corpus must contain 80 events")

    by_domain: dict[str, dict[str, object]] = {
        domain: {
            "events": 0,
            "prefixes": 0,
            "candidate_instances": 0,
            "native_action_rows": 0,
            "target_fields_in_causal_rows": set(),
            "target_fields_in_arrays": set(),
            "scenes": set(),
            "episodes": set(),
        }
        for domain in DOMAINS
    }
    run_summaries_checked = 0
    for event in population:
        domain = str(event.get("dataset"))
        if domain not in DOMAINS:
            raise ProbeAuditError(f"unknown MF3ZQ domain: {domain}")
        bucket = by_domain[domain]
        bucket["events"] = int(bucket["events"]) + 1
        cast_scenes = bucket["scenes"]
        cast_episodes = bucket["episodes"]
        assert isinstance(cast_scenes, set) and isinstance(cast_episodes, set)
        cast_scenes.add(str(event.get("scene_id")))
        cast_episodes.add(str(event.get("episode_id")))

        observation_dir = _project_path(event.get("observation_dir"))
        summary_path = observation_dir / "RUN_SUMMARY.json"
        summary = _read_object(summary_path)
        required_summary = {
            "dataset": domain,
            "episode_id": str(event.get("episode_id")),
            "split": "train",
            "no_outcome_or_target_input": True,
            "target_received": False,
            "source_target_action_compared": False,
            "public_split_access": False,
            "task_metric_payload_read": False,
        }
        for key, expected in required_summary.items():
            if summary.get(key) != expected:
                raise ProbeAuditError(f"frozen observation summary drift: {summary_path}:{key}")
        causal_info = summary.get("causal_prefix_records")
        if not isinstance(causal_info, Mapping):
            raise ProbeAuditError(f"missing causal-prefix inventory: {summary_path}")
        causal_path = _check_inventory_record(causal_info)
        causal_rows = _read_jsonl(causal_path)
        if len(causal_rows) != int(summary.get("prefix_records", -1)):
            raise ProbeAuditError(f"causal-prefix count drift: {causal_path}")
        run_summaries_checked += 1
        for row in causal_rows:
            bucket["prefixes"] = int(bucket["prefixes"]) + 1
            candidates = row.get("candidate_action_ids")
            if not isinstance(candidates, list):
                raise ProbeAuditError(f"candidate IDs missing from {causal_path}")
            bucket["candidate_instances"] = int(bucket["candidate_instances"]) + len(candidates)
            if "native_action_id" in row:
                bucket["native_action_rows"] = int(bucket["native_action_rows"]) + 1
            target_fields = bucket["target_fields_in_causal_rows"]
            assert isinstance(target_fields, set)
            target_fields.update(_target_like_fields(row.keys()))
            array_info = row.get("arrays")
            if not isinstance(array_info, Mapping):
                raise ProbeAuditError(f"array inventory missing from {causal_path}")
            array_path = _check_inventory_record(array_info)
            with np.load(array_path, allow_pickle=False) as arrays:
                array_target_fields = bucket["target_fields_in_arrays"]
                assert isinstance(array_target_fields, set)
                array_target_fields.update(_target_like_fields(arrays.files))

    sources: list[TargetSourceSummary] = []
    report: dict[str, object] = {}
    for domain in DOMAINS:
        bucket = by_domain[domain]
        row_fields = sorted(bucket.pop("target_fields_in_causal_rows"))
        array_fields = sorted(bucket.pop("target_fields_in_arrays"))
        scenes = bucket.pop("scenes")
        episodes = bucket.pop("episodes")
        assert isinstance(scenes, set) and isinstance(episodes, set)
        if row_fields or array_fields:
            raise ProbeAuditError(f"unexpected target field in frozen observation corpus: {domain}")
        source = TargetSourceSummary(
            source_id=f"mf3zq_{domain.lower()}_frozen_native_observations",
            dataset=domain,
            provenance="frozen_native_action_self_label",
            preexisting=True,
            train_development_only=True,
            public_split_accessed=False,
            exact_same_episode_prefix=True,
            exact_candidate_set_alignment=True,
            target_rows=0,
            rankable_target_rows=0,
            raw_scene_count=len(scenes),
            unique_episode_count=len(episodes),
            accepted=False,
            rejection_reasons=(
                "NO_CANDIDATE_TARGET_FIELD",
                "NATIVE_ACTION_IS_FROZEN_POLICY_PREDICTION_NOT_SUPERVISION",
            ),
        )
        source.validate()
        sources.append(source)
        report[domain] = {
            **bucket,
            "raw_scene_count": len(scenes),
            "unique_episode_count": len(episodes),
            "target_fields_in_causal_rows": row_fields,
            "target_fields_in_arrays": array_fields,
            "legal_candidate_target_rows": 0,
            "accepted": False,
            "rejection_reasons": list(source.rejection_reasons),
        }
    report["run_summaries_checked"] = run_summaries_checked
    report["source_population"] = inventory(MF3ZQ_POPULATION)
    report["observation_collection_status"] = inventory(MF3ZP_OBSERVATION_STATUS)
    return sources, report


def _audit_r2r_rejected_sources() -> tuple[list[TargetSourceSummary], dict[str, object]]:
    with gzip.open(R2R_TRAIN_GT, "rt", encoding="utf-8") as stream:
        train_gt = json.load(stream)
    if not isinstance(train_gt, dict) or not train_gt:
        raise ProbeAuditError("R2R train_gt is malformed")
    train_gt_fields: set[str] = set()
    for value in train_gt.values():
        if not isinstance(value, dict):
            raise ProbeAuditError("R2R train_gt episode is malformed")
        train_gt_fields.update(str(field) for field in value)
    candidate_target_fields = _target_like_fields(train_gt_fields)
    if candidate_target_fields:
        raise ProbeAuditError("unexpected candidate target in low-level R2R train_gt")

    # MF3ZK is inventoried only to document exclusion.  Its numerical records
    # are intentionally not parsed: the artifact is a route-return/utility
    # source by its sealed revision contract and cannot become a candidate
    # ranking target regardless of its values.
    outcome_manifest_inventory = inventory(MF3ZK_R2R_OUTCOME_MANIFEST)

    trainer_text = ETP_TRAINER.read_text(encoding="utf-8")
    if "def _teacher_action_new(" not in trainer_text or "gmap_vp_ids" not in trainer_text:
        raise ProbeAuditError("ETP runtime teacher implementation drift")

    binding_audit = _read_object(MF3ZR_BINDING_AUDIT)
    if binding_audit.get("valid_option_binding_events") != 0:
        raise ProbeAuditError("MF3ZR binding status changed")
    if binding_audit.get("binding_state_counts") != {"UNRESOLVED": 4582}:
        raise ProbeAuditError("MF3ZR binding-state audit drift")
    if binding_audit.get("public_split_access") != PUBLIC_CLOSED:
        raise ProbeAuditError("MF3ZR binding audit opened public split")

    route_source = TargetSourceSummary(
        source_id="r2r_official_low_level_train_trajectory",
        dataset="R2R",
        provenance="route_truth_reconstruction",
        preexisting=True,
        train_development_only=True,
        public_split_accessed=False,
        exact_same_episode_prefix=False,
        exact_candidate_set_alignment=False,
        target_rows=0,
        rankable_target_rows=0,
        raw_scene_count=0,
        unique_episode_count=len(train_gt),
        accepted=False,
        rejection_reasons=(
            "LOW_LEVEL_TRAJECTORY_HAS_NO_DYNAMIC_ETP_CANDIDATE_IDS",
            "POST_HOC_ROUTE_TO_CANDIDATE_MAPPING_FORBIDDEN",
        ),
    )
    outcome_source = TargetSourceSummary(
        source_id="mf3zk_r2r_exact_one_switch_route_outcomes",
        dataset="R2R",
        provenance="route_level_outcome",
        preexisting=True,
        train_development_only=True,
        public_split_accessed=False,
        exact_same_episode_prefix=True,
        exact_candidate_set_alignment=False,
        target_rows=0,
        rankable_target_rows=0,
        raw_scene_count=0,
        unique_episode_count=0,
        accepted=False,
        rejection_reasons=(
            "LABEL_IS_ROUTE_LEVEL_UTILITY_NOT_CANDIDATE_RANKING_TARGET",
        ),
    )
    for source in (route_source, outcome_source):
        source.validate()
    return [route_source, outcome_source], {
        "official_train_gt": {
            "source": inventory(R2R_TRAIN_GT),
            "episode_records": len(train_gt),
            "fields": sorted(train_gt_fields),
            "candidate_target_fields": sorted(candidate_target_fields),
            "dynamic_ETP_candidate_ids_present": False,
            "accepted": False,
            "reason": "low-level train route cannot be post-hoc mapped to a dynamic frozen candidate set",
        },
        "runtime_teacher_implementation": {
            "source": inventory(ETP_TRAINER),
            "exact_teacher_requires_runtime_dynamic_gmap_vp_ids": True,
            "accepted_preexisting_R2R_teacher_trace_in_sealed_inventory": False,
            "accepted": False,
        },
        "mf3zk_route_outcome": {
            "source": outcome_manifest_inventory,
            "sealed_source_type": "paired route return with nDTW/SDTW/SPL utility",
            "numerical_records_parsed": False,
            "accepted": False,
            "reason": "route return and utility are forbidden candidate targets",
        },
        "mf3zr_option_binding": {
            "source": inventory(MF3ZR_BINDING_AUDIT),
            "binding_edges": int(binding_audit.get("binding_edge_count", 0)),
            "binding_state_counts": binding_audit.get("binding_state_counts"),
            "valid_option_binding_events": int(binding_audit.get("valid_option_binding_events", 0)),
            "accepted_as_candidate_target": False,
        },
    }


def build_target_support_audit() -> dict[str, object]:
    """Read only sealed, train-development sources and apply the first gate."""

    protocol = verify_protocol(PROTOCOL_PATH)
    rxr_source, rxr_report = _audit_rxr_exact_targets()
    frozen_sources, frozen_report = _audit_frozen_observation_corpus()
    rejected_r2r_sources, rejected_r2r_report = _audit_r2r_rejected_sources()
    sources = [rxr_source, *frozen_sources, *rejected_r2r_sources]
    gate = evaluate_target_support(sources)
    if gate["passed"]:
        raise ProbeAuditError(
            "fixed MF3ZT source inventory unexpectedly passes; a separately reviewed population build is required"
        )

    return {
        "schema_version": "revealnav-mf3zt-decision-target-support-audit/1",
        "revision": REVISION,
        "status": gate["status"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "audit_scope": "preexisting train-development exact candidate-ranking target support",
        "source_discovery_boundary": {
            "audited_preexisting_source_classes": protocol["target_support_gate"][
                "audited_preexisting_source_classes"
            ],
            "negative_existence_claim_scope": protocol["target_support_gate"][
                "negative_existence_claim_scope"
            ],
            "new_target_generation_attempted": False,
        },
        "source_summaries": [asdict(source) for source in sources],
        "rxr_exact_target_audit": rxr_report,
        "frozen_observation_corpus_audit": frozen_report,
        "r2r_rejected_source_audit": rejected_r2r_report,
        "domain_support": gate["domain_support"],
        "first_failure": gate["first_failure"],
        "decision_population_authorized": False,
        "memory_required_classification_authorized": False,
        "evidence_memory_authorized": False,
        "reranker_or_training_authorized": False,
        "bootstrap_authorized": False,
        "public_split_access": dict(PUBLIC_CLOSED),
        "outcome_or_utility_payload_parsed": False,
        "outcome_or_utility_used_as_candidate_target": False,
        "route_truth_used_to_construct_candidate_target": False,
        "native_action_used_as_candidate_target": False,
        "conclusion": "R2R has no pre-existing exact candidate-aligned ranking target in the sealed source inventory; RxR support cannot compensate across domains.",
    }


def _atomic_write_new(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ProbeAuditError(f"immutable MF3ZT artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProbeAuditError(f"stale MF3ZT artifact partial: {partial.name}")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_target_support_audit() -> dict[str, object]:
    if TARGET_AUDIT_PATH.exists() or TARGET_AUDIT_PATH.is_symlink():
        raise ProbeAuditError("immutable MF3ZT target-support audit already exists")
    value = build_target_support_audit()
    _atomic_write_new(TARGET_AUDIT_PATH, value)
    return value


def _read_verified_target_audit() -> dict[str, object]:
    if not TARGET_AUDIT_PATH.is_file() or TARGET_AUDIT_PATH.is_symlink():
        raise ProbeAuditError("MF3ZT target-support audit is missing")
    audit = _read_object(TARGET_AUDIT_PATH)
    if audit.get("revision") != REVISION:
        raise ProbeAuditError("MF3ZT target-support revision drift")
    if audit.get("status") != "MF3ZT_DECISION_TARGET_SUPPORT_FAIL":
        raise ProbeAuditError("MF3ZT fixed support audit is not fail-closed")
    if audit.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ProbeAuditError("MF3ZT target-support protocol linkage drift")
    if audit.get("public_split_access") != PUBLIC_CLOSED:
        raise ProbeAuditError("MF3ZT target-support audit opened public split")
    domain_support = audit.get("domain_support")
    if not isinstance(domain_support, Mapping):
        raise ProbeAuditError("MF3ZT domain-support result missing")
    if domain_support.get("R2R", {}).get("legal_rankable_target_rows") != 0:
        raise ProbeAuditError("MF3ZT R2R support no longer fails")
    if audit.get("decision_population_authorized") is not False:
        raise ProbeAuditError("MF3ZT audit authorized a population after target failure")
    for key in (
        "memory_required_classification_authorized",
        "evidence_memory_authorized",
        "reranker_or_training_authorized",
        "bootstrap_authorized",
    ):
        if audit.get(key) is not False:
            raise ProbeAuditError(f"MF3ZT target failure authorized downstream work: {key}")
    for key in (
        "outcome_or_utility_payload_parsed",
        "outcome_or_utility_used_as_candidate_target",
        "route_truth_used_to_construct_candidate_target",
        "native_action_used_as_candidate_target",
    ):
        if audit.get(key) is not False:
            raise ProbeAuditError(f"MF3ZT target audit contamination flag opened: {key}")
    return audit


def build_fail_result() -> dict[str, object]:
    protocol = verify_protocol(PROTOCOL_PATH)
    audit = _read_verified_target_audit()
    for name in DOWNSTREAM_FILENAMES:
        if (OUTPUT / name).exists() or (OUTPUT / name).is_symlink():
            raise ProbeAuditError(f"downstream MF3ZT artifact exists after target failure: {name}")

    return {
        "schema_version": "revealnav-mf3zt-evidence-memory-decision-probe-result/1",
        "revision": REVISION,
        "status": "MF3ZT_DECISION_TARGET_SUPPORT_FAIL",
        "final_pass_fail": "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL",
        "first_failure": "decision_target_support",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "target_support_audit_sha256": sha256_file(TARGET_AUDIT_PATH),
        "source_commit": protocol["source_commit"],
        "population": {
            "status": "NOT_MATERIALIZED_DUE_TO_TARGET_SUPPORT_GATE",
            "sha256": None,
            "total_decisions": None,
            "R2R": None,
            "RxR": None,
            "raw_scenes": None,
        },
        "decision_target": {
            "status": "UNSUPPORTED_IN_R2R",
            "source_sha256": None,
            "domain_support": audit["domain_support"],
        },
        "memory_required": {
            "status": "NOT_CLASSIFIED",
            "counts": None,
            "scenes": None,
        },
        "evidence_memory": {
            "status": "NOT_MATERIALIZED",
            "sha256": None,
            "diagnostics": "NOT_RUN",
        },
        "arms": {
            "ETP_CURRENT": "NOT_RUN",
            "ETP_PLUS_EVIDENCE_MEMORY": "NOT_RUN",
            "ETP_PLUS_SHUFFLED_MEMORY": "NOT_RUN",
        },
        "metrics_per_domain": "NOT_RUN",
        "pairwise_deltas": "NOT_RUN",
        "scene_bootstrap_CI": "NOT_RUN",
        "evidence_diagnostics": "NOT_RUN",
        "scientific_evidence_about_memory": "NOT_OBSERVED",
        "execution": {
            "target_support_audit_run": True,
            "population_built": False,
            "memory_required_classification_run": False,
            "evidence_memory_built": False,
            "reranker_implemented": False,
            "training_started": False,
            "OOF_evaluation_run": False,
            "bootstrap_run": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
            "checkpoint_for_deployment": False,
            "public_split_access": dict(PUBLIC_CLOSED),
        },
        "public_split_access": dict(PUBLIC_CLOSED),
        "full_navigation_run": False,
        "checkpoint_generated": False,
        "interpretation": "The sealed audited source inventory lacks legal R2R candidate-ranking target support. No evidence-memory hypothesis was tested, so this is not evidence for or against evidence memory.",
        "stop_rule": {
            "triggered": True,
            "reason": "MF3ZT_DECISION_TARGET_SUPPORT_FAIL",
            "automatic_rescue_forbidden": True,
            "next_action": "STOP",
        },
    }


def write_fail_result() -> dict[str, object]:
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise ProbeAuditError("immutable MF3ZT result already exists")
    value = build_fail_result()
    _atomic_write_new(RESULT_PATH, value)
    return value


def verify_result() -> dict[str, object]:
    verify_protocol(PROTOCOL_PATH)
    audit = _read_verified_target_audit()
    if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink():
        raise ProbeAuditError("MF3ZT result is missing")
    result = _read_object(RESULT_PATH)
    if result.get("status") != "MF3ZT_DECISION_TARGET_SUPPORT_FAIL":
        raise ProbeAuditError("MF3ZT result status drift")
    if result.get("final_pass_fail") != "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL":
        raise ProbeAuditError("MF3ZT final decision drift")
    if result.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ProbeAuditError("MF3ZT result protocol linkage drift")
    if result.get("target_support_audit_sha256") != sha256_file(TARGET_AUDIT_PATH):
        raise ProbeAuditError("MF3ZT result audit linkage drift")
    if result.get("decision_target", {}).get("domain_support") != audit.get("domain_support"):
        raise ProbeAuditError("MF3ZT result target-support summary drift")
    if result.get("public_split_access") != PUBLIC_CLOSED:
        raise ProbeAuditError("MF3ZT result opened public split")
    if result.get("scientific_evidence_about_memory") != "NOT_OBSERVED":
        raise ProbeAuditError("MF3ZT target failure claims unrun memory evidence")
    if result.get("metrics_per_domain") != "NOT_RUN" or result.get("scene_bootstrap_CI") != "NOT_RUN":
        raise ProbeAuditError("MF3ZT target failure contains downstream metrics")
    if result.get("arms") != {
        "ETP_CURRENT": "NOT_RUN",
        "ETP_PLUS_EVIDENCE_MEMORY": "NOT_RUN",
        "ETP_PLUS_SHUFFLED_MEMORY": "NOT_RUN",
    }:
        raise ProbeAuditError("MF3ZT target failure arm status drift")
    execution = result.get("execution")
    if not isinstance(execution, Mapping):
        raise ProbeAuditError("MF3ZT result execution record missing")
    if execution.get("public_split_access") != PUBLIC_CLOSED:
        raise ProbeAuditError("MF3ZT result execution opened public split")
    for key in (
        "population_built",
        "memory_required_classification_run",
        "evidence_memory_built",
        "reranker_implemented",
        "training_started",
        "OOF_evaluation_run",
        "bootstrap_run",
        "full_navigation_run",
        "checkpoint_generated",
        "checkpoint_for_deployment",
    ):
        if execution.get(key) is not False:
            raise ProbeAuditError(f"MF3ZT downstream execution occurred: {key}")
    if result.get("full_navigation_run") is not False:
        raise ProbeAuditError("MF3ZT result ran full navigation")
    if result.get("checkpoint_generated") is not False:
        raise ProbeAuditError("MF3ZT result generated a checkpoint")
    for name in DOWNSTREAM_FILENAMES:
        if (OUTPUT / name).exists() or (OUTPUT / name).is_symlink():
            raise ProbeAuditError(f"downstream MF3ZT artifact exists after target failure: {name}")
    return result


__all__ = [
    "TARGET_FIELD_NAMES",
    "ALLOWED_TARGET_PROVENANCE",
    "FORBIDDEN_TARGET_PROVENANCE",
    "DOWNSTREAM_FILENAMES",
    "ProbeAuditError",
    "TargetSourceSummary",
    "evaluate_target_support",
    "build_target_support_audit",
    "write_target_support_audit",
    "build_fail_result",
    "write_fail_result",
    "verify_result",
]
