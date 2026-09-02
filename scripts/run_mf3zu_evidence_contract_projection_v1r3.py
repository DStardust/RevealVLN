#!/usr/bin/env python3
"""Final target-blind projection of unsupported v1r2 historical evidence."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import annotate_mf3zu_rxr_evidence as annotation  # noqa: E402
import build_mf3zu_evidence_memory as memory_builder  # noqa: E402
import run_mf3zu_annotation_recovery_v1r1 as v1r1  # noqa: E402
import run_mf3zu_evidence_contract_recovery_v1r2 as v1r2  # noqa: E402
from revealnav_mf3.mf3zu_evidence_memory import (  # noqa: E402
    ConfidenceClass,
    REVISION as SCIENTIFIC_REVISION,
    stable_sha256,
    validate_evidence_response,
)


REVISION = "mf3zu_rxr_evidence_annotation_conservative_projection_v1r3"
SEALED_STATUS = "SEALED_BEFORE_MF3ZU_V1R3_PROJECTED_RESPONSES"
EXPECTED_PARENT_PASS = 1416
EXPECTED_PARENT_FAIL = 12
EXPECTED_PROJECTED_ATOMS = 20
PARENT_OUTPUT = v1r2.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_annotation_projection_v1r3"
)
METHOD = (
    ROOT / "METHOD_REVISION_3ZU_RXR_EVIDENCE_CONSERVATIVE_PROJECTION_V1R3.md"
)
TEST = ROOT / "tests/test_evidence_contract_projection_mf3zu_v1r3.py"
PROTOCOL_NAME = "MF3ZU_V1R3_EVIDENCE_PROJECTION_PROTOCOL.json"
INPUT_LEDGER_NAME = "MF3ZU_V1R3_PROJECTION_INPUT_LEDGER.jsonl"
OUTPUT_LEDGER_NAME = "MF3ZU_V1R3_PROJECTION_OUTPUT_LEDGER.jsonl"
STATUS_NAME = "MF3ZU_V1R3_EVIDENCE_PROJECTION_STATUS.json"
FREEZE_PROVENANCE_NAME = "MF3ZU_V1R3_EVIDENCE_FREEZE_PROVENANCE.json"
SUPPORT_AUDIT_NAME = "MF3ZU_V1R3_PRETRAIN_SUPPORT_AUDIT.json"
_WRITE_LOCK = threading.Lock()


class V1R3Error(RuntimeError):
    """Raised on any unsupported projection or provenance drift."""


sha256_file = v1r2.sha256_file
rel = v1r2.rel
inventory = v1r2.inventory
strict_json = v1r2.strict_json
jsonl = v1r2.jsonl
atomic_json = v1r2.atomic_json
atomic_jsonl = v1r2.atomic_jsonl
atomic_copy = v1r2.atomic_copy
stage_lock = v1r2.stage_lock
_json_bytes = v1r2._json_bytes
_jsonl_bytes = v1r2._jsonl_bytes


def project_unsupported_history(
    value: object,
    *,
    expected_atom_ids: Sequence[str],
    decision_step: int,
    allowed_candidate_ids: Sequence[str],
    graph: object,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Suppress unsupported historical positives without inventing evidence."""

    rows, structural_issues = v1r2._structural_rows(
        value, expected_atom_ids=expected_atom_ids
    )
    if structural_issues:
        raise V1R3Error("v1r3 projection cannot repair a schema violation")
    projected = copy.deepcopy({"atoms": rows})
    operations: list[str] = []
    for row in projected["atoms"]:
        source = row["source_step"]
        if row["historical_status"] == ConfidenceClass.OBSERVED.value and (
            isinstance(source, bool)
            or not isinstance(source, int)
            or not 0 <= source < decision_step
        ):
            row["historical_status"] = ConfidenceClass.AMBIGUOUS.value
            row["source_step"] = None
            operations.append(
                f"PROJECT_UNSUPPORTED_HISTORY:{row['instruction_atom_id']}"
            )
    by_id = {str(row["instruction_atom_id"]): row for row in projected["atoms"]}
    if set(by_id) != set(expected_atom_ids) or len(by_id) != len(expected_atom_ids):
        raise V1R3Error("v1r3 projected atom identity drift")
    projected = {"atoms": [by_id[value] for value in expected_atom_ids]}
    validate_evidence_response(
        projected,
        graph=graph,
        decision_step=decision_step,
        allowed_candidate_ids=allowed_candidate_ids,
    )
    if not operations:
        # Idempotent verification is allowed only for an already valid value.
        validate_evidence_response(
            value,
            graph=graph,
            decision_step=decision_step,
            allowed_candidate_ids=allowed_candidate_ids,
        )
    return projected, tuple(operations)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_implementation_at_commit(
    commit: str,
    implementation: Mapping[str, object],
) -> None:
    v1r2._verify_implementation_at_commit(commit, implementation)


def _parent_bundle() -> dict[str, object]:
    parent_protocol = strict_json(PARENT_OUTPUT / v1r2.PROTOCOL_NAME)
    if (
        parent_protocol.get("revision") != v1r2.REVISION
        or parent_protocol.get("status") != v1r2.SEALED_STATUS
    ):
        raise V1R3Error("v1r2 protocol identity drift")
    input_manifest = strict_json(
        PARENT_OUTPUT / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
    )
    if (
        input_manifest.get("revision") != SCIENTIFIC_REVISION
        or input_manifest.get("status") != "SEALED_BEFORE_EVIDENCE_RESPONSES"
        or input_manifest.get("request_count") != 1428
    ):
        raise V1R3Error("v1r2 evidence input boundary drift")
    request_info = input_manifest.get("requests")
    if not isinstance(request_info, Mapping):
        raise V1R3Error("v1r2 evidence request inventory is missing")
    request_path = ROOT / str(request_info.get("path", ""))
    if inventory(request_path) != dict(request_info):
        raise V1R3Error("sealed evidence requests changed")
    requests = jsonl(request_path)
    if len(requests) != 1428:
        raise V1R3Error("evidence request count drift")

    manifest_path = PARENT_OUTPUT / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("revision") != SCIENTIFIC_REVISION
        or manifest.get("contract_repair_revision") != v1r2.REVISION
        or manifest.get("status") != "FAIL"
        or manifest.get("planned") != 1428
        or manifest.get("response_files") != 1428
        or manifest.get("pass") != EXPECTED_PARENT_PASS
        or manifest.get("fail") != EXPECTED_PARENT_FAIL
        or manifest.get("ranking_label_read") is not False
        or manifest.get("task_metric_read") is not False
        or manifest.get("public_split_access") is not False
        or manifest.get("human_verified") is not False
        or manifest.get("gold") is not False
    ):
        raise V1R3Error("v1r2 annotation result drift")
    failures = manifest.get("failures")
    if not isinstance(failures, list) or len(failures) != EXPECTED_PARENT_FAIL:
        raise V1R3Error("v1r2 failure list drift")
    failure_ids = {
        str(row.get("request_id")) for row in failures
        if isinstance(row, Mapping)
    }
    if len(failure_ids) != EXPECTED_PARENT_FAIL:
        raise V1R3Error("v1r2 failure identity drift")

    graphs, _ = v1r1._recovery_instruction_graphs(v1r1.DEFAULT_OUTPUT)
    records: list[dict[str, object]] = []
    response_inventories: list[dict[str, object]] = []
    passed = failed = projected_atoms = 0
    for request in requests:
        request_id = str(request["request_id"])
        graph = graphs[str(request["episode_id"])]
        response_path = (
            PARENT_OUTPUT / "responses/evidence" / f"{request_id}.json"
        )
        response_info = inventory(response_path)
        response_inventories.append(response_info)
        response = strict_json(response_path)
        if (
            response.get("revision") != SCIENTIFIC_REVISION
            or response.get("request_id") != request_id
            or response.get("event_id") != str(request["event_id"])
            or response.get("ranking_label_read") is not False
            or response.get("task_metric_read") is not False
            or response.get("public_split_access") is not False
            or response.get("human_verified") is not False
            or response.get("gold") is not False
        ):
            raise V1R3Error("v1r2 response provenance drift")
        if response.get("status") == "PASS":
            if response.get("contract_repair_revision") not in {
                v1r1.REVISION,
                v1r2.REVISION,
            }:
                raise V1R3Error("v1r2 PASS repair revision drift")
            if request_id in failure_ids:
                raise V1R3Error("v1r2 PASS is listed as failure")
            validate_evidence_response(
                response.get("response"),
                graph=graph,
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
            )
            mode = "PARENT_PASS_BYTE_REUSE"
            passed += 1
            operations: tuple[str, ...] = ()
        elif response.get("status") == "FAIL":
            if response.get("contract_repair_revision") != v1r2.REVISION:
                raise V1R3Error("v1r2 FAIL repair revision drift")
            if request_id not in failure_ids or response.get("error") != (
                "MF3ZUContractError: historical source must be strictly before the decision"
            ):
                raise V1R3Error("v1r2 failure is outside projection support")
            projected, operations = project_unsupported_history(
                response.get("invalid_parsed_response"),
                expected_atom_ids=v1r2._expected_atom_ids(request),
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
                graph=graph,
            )
            del projected
            if not operations:
                raise V1R3Error("v1r2 failure has no projectable atom")
            projected_atoms += len(operations)
            mode = "CONSERVATIVE_UNSUPPORTED_HISTORY_PROJECTION"
            failed += 1
        else:
            raise V1R3Error("v1r2 response status drift")
        records.append({
            "request": request,
            "graph": graph,
            "response": response,
            "response_inventory": response_info,
            "mode": mode,
            "operations": operations,
        })
    if (passed, failed, projected_atoms) != (
        EXPECTED_PARENT_PASS,
        EXPECTED_PARENT_FAIL,
        EXPECTED_PROJECTED_ATOMS,
    ):
        raise V1R3Error("v1r3 fixed projection partition drift")
    bundle_sha256 = stable_sha256(response_inventories)
    if manifest.get("response_bundle_sha256") != bundle_sha256:
        raise V1R3Error("v1r2 response bundle SHA drift")
    return {
        "input_manifest": input_manifest,
        "manifest": manifest,
        "requests": requests,
        "graphs": graphs,
        "records": records,
        "response_bundle_sha256": bundle_sha256,
    }


def _input_ledger_rows(bundle: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in bundle["records"]:
        if record["mode"] == "PARENT_PASS_BYTE_REUSE":
            continue
        request = record["request"]
        raw = record["response"]["invalid_parsed_response"]
        projected_details = []
        for atom in raw["atoms"]:
            source = atom["source_step"]
            if atom["historical_status"] == ConfidenceClass.OBSERVED.value and (
                isinstance(source, bool)
                or not isinstance(source, int)
                or not 0 <= source < int(request["decision_step"])
            ):
                projected_details.append({
                    "instruction_atom_id": str(atom["instruction_atom_id"]),
                    "before_historical_status": ConfidenceClass.OBSERVED.value,
                    "before_source_step": source,
                    "after_historical_status": ConfidenceClass.AMBIGUOUS.value,
                    "after_source_step": None,
                    "semantic_value_role_after_projection": (
                        "retained_unverified_audit_text_not_memory"
                    ),
                })
        rows.append({
            "schema_version": "revealnav-mf3zu-v1r3-projection-input/1",
            "revision": REVISION,
            "request_id": str(request["request_id"]),
            "event_id": str(request["event_id"]),
            "decision_step": int(request["decision_step"]),
            "projected_atoms": list(record["operations"]),
            "projected_atom_details": projected_details,
            "source_parent_response": dict(record["response_inventory"]),
            "candidate_target_accessed": False,
            "performance_accessed": False,
            "public_split_access": False,
        })
    rows.sort(key=lambda row: str(row["request_id"]))
    if len(rows) != EXPECTED_PARENT_FAIL or sum(
        len(row["projected_atoms"]) for row in rows
    ) != EXPECTED_PROJECTED_ATOMS:
        raise V1R3Error("v1r3 projection input ledger drift")
    return rows


def _implementation_inventory() -> dict[str, object]:
    return {
        "method": inventory(METHOD),
        "runner": inventory(Path(__file__).resolve()),
        "regression_test": inventory(TEST),
    }


def _protocol_value(
    *,
    bundle: Mapping[str, object],
    source_commit: str,
    output: Path,
) -> dict[str, object]:
    return {
        "schema_version": "revealnav-mf3zu-v1r3-evidence-projection-protocol/1",
        "revision": REVISION,
        "status": SEALED_STATUS,
        "source_commit": source_commit,
        "implementation": _implementation_inventory(),
        "parent": {
            "v1r2_method": inventory(v1r2.METHOD),
            "v1r2_protocol": inventory(PARENT_OUTPUT / v1r2.PROTOCOL_NAME),
            "v1r2_annotation_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
            ),
            "v1r2_status": inventory(PARENT_OUTPUT / v1r2.STATUS_NAME),
            "evidence_input_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
            ),
            "evidence_requests": dict(bundle["input_manifest"])["requests"],
            "response_bundle_sha256": bundle["response_bundle_sha256"],
            "response_files": 1428,
            "pass": EXPECTED_PARENT_PASS,
            "fail": EXPECTED_PARENT_FAIL,
            "files_read_only": True,
        },
        "projection_input_ledger": inventory(output / INPUT_LEDGER_NAME),
        "fixed_projection": {
            "parent_pass_byte_reuse": EXPECTED_PARENT_PASS,
            "projected_responses": EXPECTED_PARENT_FAIL,
            "projected_atoms": EXPECTED_PROJECTED_ATOMS,
            "predicate": (
                "historical_status==OBSERVED and source_step not integer in "
                "[0,decision_step)"
            ),
            "operation": (
                "historical_status:=AMBIGUOUS; source_step:=null"
            ),
            "other_fields_changed": False,
            "provider_calls": 0,
            "history_step_clamped_or_fabricated": False,
            "projected_atom_can_enter_memory": False,
            "selection_uses_support_or_performance": False,
            "validator_change": False,
        },
        "inherited_support_gate": {
            "minimum_decisions": 50,
            "minimum_raw_scenes": 10,
            "memory_required_definition_changed": False,
            "support_failure_stops_before_training": True,
        },
        "output_root": rel(output),
        "boundary": {
            "candidate_target_accessed": False,
            "performance_accessed": False,
            "outcome_or_utility_accessed": False,
            "future_observation_accessed": False,
            "public_split_access": {
                "val_seen": False,
                "val_unseen": False,
                "test": False,
                "test_challenge": False,
            },
            "human_verified": False,
            "gold": False,
            "training_authorized_by_this_revision": False,
            "training_run": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
        },
    }


def seal(output: Path) -> dict[str, object]:
    with stage_lock(output, "seal"):
        protocol_path = output / PROTOCOL_NAME
        if protocol_path.is_file() and not protocol_path.is_symlink():
            return verify_protocol(output)
        for forbidden in (
            output / "responses/evidence",
            output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json",
            output / "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl",
        ):
            if forbidden.exists() or forbidden.is_symlink():
                raise V1R3Error("projected output exists before protocol seal")
        bundle = _parent_bundle()
        source_commit = _git_head()
        implementation = _implementation_inventory()
        _verify_implementation_at_commit(source_commit, implementation)
        output.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(output / INPUT_LEDGER_NAME, _input_ledger_rows(bundle))
        value = _protocol_value(
            bundle=bundle, source_commit=source_commit, output=output
        )
        atomic_json(protocol_path, value)
        return value


def verify_protocol(output: Path) -> dict[str, object]:
    value = strict_json(output / PROTOCOL_NAME)
    if value.get("revision") != REVISION or value.get("status") != SEALED_STATUS:
        raise V1R3Error("v1r3 protocol identity drift")
    bundle = _parent_bundle()
    expected_ledger = _jsonl_bytes(_input_ledger_rows(bundle))
    if (output / INPUT_LEDGER_NAME).read_bytes() != expected_ledger:
        raise V1R3Error("v1r3 projection input ledger drift")
    expected = _protocol_value(
        bundle=bundle,
        source_commit=str(value.get("source_commit", "")),
        output=output,
    )
    if value != expected:
        raise V1R3Error("v1r3 protocol/source drift")
    _verify_implementation_at_commit(
        str(value["source_commit"]), value["implementation"]
    )
    return value


def _materialize_parent_views(output: Path) -> None:
    for name in (
        "MF3ZU_EVIDENCE_INPUT_MANIFEST.json",
        "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json",
        "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json",
        "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json",
        "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json",
    ):
        atomic_copy(PARENT_OUTPUT / name, output / name)


def _projected_response_value(
    *,
    record: Mapping[str, object],
    protocol_sha256: str,
) -> dict[str, object]:
    request = record["request"]
    response = record["response"]
    projected, operations = project_unsupported_history(
        response.get("invalid_parsed_response"),
        expected_atom_ids=v1r2._expected_atom_ids(request),
        decision_step=int(request["decision_step"]),
        allowed_candidate_ids=list(request["candidate_alias_to_action_id"]),
        graph=record["graph"],
    )
    if operations != record["operations"] or not operations:
        raise V1R3Error("projected operation ledger drift")
    return {
        "schema_version": "revealnav-mf3zu-v1r3-evidence-response/1",
        "revision": SCIENTIFIC_REVISION,
        "contract_repair_revision": REVISION,
        "status": "PASS",
        "stage": "evidence_conservative_projection",
        "request_id": str(request["request_id"]),
        "event_id": str(request["event_id"]),
        "response": projected,
        "projection_operations": list(operations),
        "source_parent_response": dict(record["response_inventory"]),
        "projection_protocol_sha256": protocol_sha256,
        "provider_call_performed": False,
        "projected_history_can_enter_memory": False,
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }


def _validate_output_response(
    *,
    record: Mapping[str, object],
    output: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    request = record["request"]
    destination = (
        output / "responses/evidence" / f"{request['request_id']}.json"
    )
    value = strict_json(destination)
    if record["mode"] == "PARENT_PASS_BYTE_REUSE":
        source = (
            PARENT_OUTPUT
            / "responses/evidence"
            / f"{request['request_id']}.json"
        )
        if destination.read_bytes() != source.read_bytes():
            raise V1R3Error("v1r2 PASS response was not byte-preserved")
        validate_evidence_response(
            value.get("response"),
            graph=record["graph"],
            decision_step=int(request["decision_step"]),
            allowed_candidate_ids=list(request["candidate_alias_to_action_id"]),
        )
        return value
    expected = _projected_response_value(
        record=record, protocol_sha256=protocol_sha256
    )
    if value != expected:
        raise V1R3Error("v1r3 projected response drift")
    return value


def _write_status(output: Path, value: Mapping[str, object]) -> None:
    with _WRITE_LOCK:
        path = output / STATUS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        partial.write_bytes(_json_bytes(dict(value)))
        os.replace(partial, path)


def materialize(output: Path) -> dict[str, object]:
    with stage_lock(output, "materialize"):
        verify_protocol(output)
        bundle = _parent_bundle()
        _materialize_parent_views(output)
        protocol_sha256 = sha256_file(output / PROTOCOL_NAME)
        response_dir = output / "responses/evidence"
        response_dir.mkdir(parents=True, exist_ok=True)
        output_ledger: list[dict[str, object]] = []
        response_inventories: list[dict[str, object]] = []
        for record in bundle["records"]:
            request = record["request"]
            destination = response_dir / f"{request['request_id']}.json"
            if record["mode"] == "PARENT_PASS_BYTE_REUSE":
                atomic_copy(
                    PARENT_OUTPUT
                    / "responses/evidence"
                    / f"{request['request_id']}.json",
                    destination,
                )
            else:
                atomic_json(
                    destination,
                    _projected_response_value(
                        record=record, protocol_sha256=protocol_sha256
                    ),
                )
            value = _validate_output_response(
                record=record,
                output=output,
                protocol_sha256=protocol_sha256,
            )
            if value.get("status") != "PASS":
                raise V1R3Error("non-passing response entered v1r3 bundle")
            response_info = inventory(destination)
            response_inventories.append(response_info)
            if record["mode"] != "PARENT_PASS_BYTE_REUSE":
                output_ledger.append({
                    "schema_version": "revealnav-mf3zu-v1r3-projection-output/1",
                    "revision": REVISION,
                    "request_id": str(request["request_id"]),
                    "event_id": str(request["event_id"]),
                    "projection_operations": list(record["operations"]),
                    "source_parent_response": dict(
                        record["response_inventory"]
                    ),
                    "projected_response": response_info,
                    "provider_call_performed": False,
                    "candidate_target_accessed": False,
                    "performance_accessed": False,
                    "public_split_access": False,
                })
        output_ledger.sort(key=lambda row: str(row["request_id"]))
        atomic_jsonl(output / OUTPUT_LEDGER_NAME, output_ledger)
        manifest = {
            "schema_version": "revealnav-mf3zu-evidence-annotation/1",
            "revision": SCIENTIFIC_REVISION,
            "contract_repair_revision": REVISION,
            "status": "PASS",
            "planned": 1428,
            "response_files": 1428,
            "pass": 1428,
            "fail": 0,
            "failures": [],
            "parent_pass_byte_reuse": EXPECTED_PARENT_PASS,
            "projected_responses": EXPECTED_PARENT_FAIL,
            "projected_atoms": EXPECTED_PROJECTED_ATOMS,
            "provider_calls": 0,
            "model": "qwen3.8-max",
            "response_bundle_sha256": stable_sha256(response_inventories),
            "source_parent_evidence_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
            ),
            "source_evidence_input_manifest": inventory(
                output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
            ),
            "projection_input_ledger": inventory(output / INPUT_LEDGER_NAME),
            "projection_output_ledger": inventory(output / OUTPUT_LEDGER_NAME),
            "projection_protocol": inventory(output / PROTOCOL_NAME),
            "ranking_label_read": False,
            "task_metric_read": False,
            "public_split_access": False,
            "human_verified": False,
            "gold": False,
        }
        atomic_json(
            output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json", manifest
        )
        _write_status(output, {
            "revision": REVISION,
            "stage": "EVIDENCE_ANNOTATION_COMPLETE",
            "planned": 1428,
            "pass": 1428,
            "fail": 0,
            "projected_responses": EXPECTED_PARENT_FAIL,
            "projected_atoms": EXPECTED_PROJECTED_ATOMS,
            "training_started": False,
        })
        return manifest


def audit_complete_bundle(output: Path) -> dict[str, object]:
    verify_protocol(output)
    bundle = _parent_bundle()
    protocol_sha256 = sha256_file(output / PROTOCOL_NAME)
    manifest_path = output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("revision") != SCIENTIFIC_REVISION
        or manifest.get("contract_repair_revision") != REVISION
        or manifest.get("status") != "PASS"
        or manifest.get("planned") != 1428
        or manifest.get("response_files") != 1428
        or manifest.get("pass") != 1428
        or manifest.get("fail") != 0
        or manifest.get("failures") != []
        or manifest.get("parent_pass_byte_reuse") != EXPECTED_PARENT_PASS
        or manifest.get("projected_responses") != EXPECTED_PARENT_FAIL
        or manifest.get("projected_atoms") != EXPECTED_PROJECTED_ATOMS
        or manifest.get("provider_calls") != 0
        or manifest.get("ranking_label_read") is not False
        or manifest.get("task_metric_read") is not False
        or manifest.get("public_split_access") is not False
        or manifest.get("human_verified") is not False
        or manifest.get("gold") is not False
    ):
        raise V1R3Error("complete v1r3 response manifest is required")
    if manifest.get("projection_protocol") != inventory(output / PROTOCOL_NAME):
        raise V1R3Error("v1r3 manifest protocol provenance drift")
    if manifest.get("projection_input_ledger") != inventory(
        output / INPUT_LEDGER_NAME
    ):
        raise V1R3Error("v1r3 input-ledger provenance drift")
    if manifest.get("source_parent_evidence_manifest") != inventory(
        PARENT_OUTPUT / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
    ):
        raise V1R3Error("v1r3 parent provenance drift")
    inventories: list[dict[str, object]] = []
    output_ledger: list[dict[str, object]] = []
    for record in bundle["records"]:
        request = record["request"]
        value = _validate_output_response(
            record=record,
            output=output,
            protocol_sha256=protocol_sha256,
        )
        if value.get("status") != "PASS":
            raise V1R3Error("non-passing response entered audited bundle")
        response_path = (
            output / "responses/evidence" / f"{request['request_id']}.json"
        )
        response_info = inventory(response_path)
        inventories.append(response_info)
        if record["mode"] != "PARENT_PASS_BYTE_REUSE":
            output_ledger.append({
                "schema_version": "revealnav-mf3zu-v1r3-projection-output/1",
                "revision": REVISION,
                "request_id": str(request["request_id"]),
                "event_id": str(request["event_id"]),
                "projection_operations": list(record["operations"]),
                "source_parent_response": dict(record["response_inventory"]),
                "projected_response": response_info,
                "provider_call_performed": False,
                "candidate_target_accessed": False,
                "performance_accessed": False,
                "public_split_access": False,
            })
    output_ledger.sort(key=lambda row: str(row["request_id"]))
    if (output / OUTPUT_LEDGER_NAME).read_bytes() != _jsonl_bytes(output_ledger):
        raise V1R3Error("v1r3 projection output ledger drift")
    if manifest.get("projection_output_ledger") != inventory(
        output / OUTPUT_LEDGER_NAME
    ):
        raise V1R3Error("v1r3 output-ledger provenance drift")
    bundle_sha256 = stable_sha256(inventories)
    if manifest.get("response_bundle_sha256") != bundle_sha256:
        raise V1R3Error("v1r3 response bundle SHA drift")
    response_files = list((output / "responses/evidence").glob("*.json"))
    if len(response_files) != 1428:
        raise V1R3Error("v1r3 response-directory coverage drift")
    return {
        "responses": 1428,
        "parent_pass_byte_reuse": EXPECTED_PARENT_PASS,
        "projected_responses": EXPECTED_PARENT_FAIL,
        "projected_atoms": EXPECTED_PROJECTED_ATOMS,
        "response_bundle_sha256": bundle_sha256,
        "manifest": inventory(manifest_path),
        "projection_output_ledger": inventory(output / OUTPUT_LEDGER_NAME),
        "candidate_target_accessed": False,
        "performance_accessed": False,
        "public_split_access": False,
    }


def _builder_instruction_graphs(
    _output: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    return v1r1._recovery_instruction_graphs(v1r1.DEFAULT_OUTPUT)


def freeze(output: Path) -> dict[str, object]:
    with stage_lock(output, "freeze"):
        bundle_audit = audit_complete_bundle(output)
        memory_path = output / "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl"
        memory_manifest_path = output / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json"
        if memory_path.exists() or memory_manifest_path.exists():
            return verify_freeze(output)
        _materialize_parent_views(output)
        if memory_builder.annotation is not annotation:
            raise V1R3Error("memory builder annotation-module identity drift")
        original_graph_loader = annotation._instruction_graphs
        original_atomic_json = memory_builder.atomic_json
        original_atomic_jsonl = memory_builder.atomic_jsonl
        annotation._instruction_graphs = _builder_instruction_graphs
        memory_builder.atomic_json = atomic_json
        memory_builder.atomic_jsonl = atomic_jsonl
        try:
            memory_manifest = memory_builder.build(output)
        finally:
            annotation._instruction_graphs = original_graph_loader
            memory_builder.atomic_json = original_atomic_json
            memory_builder.atomic_jsonl = original_atomic_jsonl
        support = memory_manifest.get("memory_required_support")
        if not isinstance(support, Mapping):
            raise V1R3Error("memory-required support record is missing")
        support_pass = support.get("pass") is True
        support_audit = {
            "schema_version": "revealnav-mf3zu-v1r3-pretrain-support-audit/1",
            "revision": REVISION,
            "status": (
                "MF3ZU_V1R3_PRETRAIN_SUPPORT_PASS"
                if support_pass
                else "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL"
            ),
            "memory_manifest": inventory(memory_manifest_path),
            "decisions": memory_manifest.get("rows"),
            "episodes": memory_manifest.get("episodes"),
            "raw_scenes": memory_manifest.get("raw_scenes"),
            "minimum_memory_required_decisions": support.get(
                "minimum_decisions"
            ),
            "minimum_memory_required_raw_scenes": support.get(
                "minimum_raw_scenes"
            ),
            "observed_memory_required_decisions": support.get(
                "observed_decisions"
            ),
            "observed_memory_required_raw_scenes": support.get(
                "observed_raw_scenes"
            ),
            "support_pass": support_pass,
            "candidate_target_accessed": False,
            "performance_accessed": False,
            "public_split_access": False,
            "training_started": False,
            "training_authorized_by_this_revision": False,
        }
        atomic_json(output / SUPPORT_AUDIT_NAME, support_audit)
        provenance = {
            "schema_version": "revealnav-mf3zu-v1r3-evidence-freeze/1",
            "revision": REVISION,
            "status": str(memory_manifest.get("status")),
            "source_protocol": inventory(output / PROTOCOL_NAME),
            "source_evidence_manifest": inventory(
                output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
            ),
            "response_bundle_audit": bundle_audit,
            "evidence_memory": inventory(memory_path),
            "evidence_memory_manifest": inventory(memory_manifest_path),
            "pretrain_support_audit": inventory(output / SUPPORT_AUDIT_NAME),
            "candidate_target_accessed": False,
            "performance_accessed": False,
            "outcome_or_utility_accessed": False,
            "public_split_access": False,
            "training_run": False,
            "training_authorized_by_this_revision": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
        }
        atomic_json(output / FREEZE_PROVENANCE_NAME, provenance)
        _write_status(output, {
            "revision": REVISION,
            "stage": (
                "EVIDENCE_FROZEN_SUPPORT_PASS"
                if support_pass
                else "MEMORY_REQUIRED_SUPPORT_FAIL"
            ),
            "annotation_pass": 1428,
            "memory_status": memory_manifest.get("status"),
            "memory_required_decisions": support.get("observed_decisions"),
            "memory_required_raw_scenes": support.get("observed_raw_scenes"),
            "training_started": False,
        })
        return provenance


def verify_freeze(output: Path) -> dict[str, object]:
    bundle_audit = audit_complete_bundle(output)
    memory_path = output / "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl"
    manifest_path = output / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("revision") != SCIENTIFIC_REVISION
        or manifest.get("rows") != 1428
        or manifest.get("episodes") != 154
        or manifest.get("raw_scenes") != 59
        or manifest.get("candidate_target_accessed") is not False
        or manifest.get("outcome_or_utility_accessed") is not False
        or manifest.get("exact_target_artifact_opened") is not False
        or manifest.get("public_split_access") is not False
        or manifest.get("full_navigation_run") is not False
        or manifest.get("checkpoint_generated") is not False
    ):
        raise V1R3Error("frozen memory manifest boundary drift")
    if manifest.get("evidence_memory") != inventory(memory_path):
        raise V1R3Error("frozen evidence memory inventory drift")
    support = manifest.get("memory_required_support")
    if not isinstance(support, Mapping):
        raise V1R3Error("frozen memory support record is missing")
    support_pass = support.get("pass") is True
    expected_status = (
        "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN"
        if support_pass
        else "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL"
    )
    if manifest.get("status") != expected_status:
        raise V1R3Error("frozen memory/support status mismatch")
    rows = jsonl(memory_path)
    if (
        len(rows) != 1428
        or len({str(row.get("event_id")) for row in rows}) != 1428
        or any(
            row.get("exact_target_artifact_opened") is not False
            or row.get("ranking_label_read") is not False
            or row.get("task_metric_read") is not False
            or row.get("public_split_access") is not False
            or int(row.get("retrieval_budget", -1)) != 8
            for row in rows
        )
    ):
        raise V1R3Error("frozen evidence memory row boundary drift")
    required = [row for row in rows if row.get("memory_required") is True]
    if (
        len(required) != int(support.get("observed_decisions", -1))
        or len({str(row["scene_id"]) for row in required})
        != int(support.get("observed_raw_scenes", -1))
    ):
        raise V1R3Error("frozen memory support count drift")
    by_event = {str(row["event_id"]): row for row in rows}
    for ledger_row in jsonl(output / INPUT_LEDGER_NAME):
        event_id = str(ledger_row["event_id"])
        memory_row = by_event.get(event_id)
        if memory_row is None or memory_row.get("memory_required") is not False:
            raise V1R3Error("projected event subgroup membership drift")
        atom_ids = {
            str(value["instruction_atom_id"])
            for value in ledger_row["projected_atom_details"]
        }
        for field in ("records", "retrieved_records"):
            values = memory_row.get(field)
            if not isinstance(values, list) or any(
                str(value.get("instruction_atom_id")) in atom_ids
                for value in values
                if isinstance(value, Mapping)
            ):
                raise V1R3Error("projected history entered evidence memory")
    support_audit = strict_json(output / SUPPORT_AUDIT_NAME)
    if (
        support_audit.get("support_pass") is not support_pass
        or support_audit.get("observed_memory_required_decisions")
        != support.get("observed_decisions")
        or support_audit.get("observed_memory_required_raw_scenes")
        != support.get("observed_raw_scenes")
        or support_audit.get("candidate_target_accessed") is not False
        or support_audit.get("performance_accessed") is not False
        or support_audit.get("training_started") is not False
    ):
        raise V1R3Error("pretrain support audit drift")
    provenance = strict_json(output / FREEZE_PROVENANCE_NAME)
    if (
        provenance.get("status") != expected_status
        or provenance.get("response_bundle_audit") != bundle_audit
        or provenance.get("evidence_memory") != inventory(memory_path)
        or provenance.get("evidence_memory_manifest") != inventory(manifest_path)
        or provenance.get("pretrain_support_audit")
        != inventory(output / SUPPORT_AUDIT_NAME)
        or provenance.get("candidate_target_accessed") is not False
        or provenance.get("performance_accessed") is not False
        or provenance.get("public_split_access") is not False
        or provenance.get("training_run") is not False
        or provenance.get("training_authorized_by_this_revision") is not False
    ):
        raise V1R3Error("v1r3 evidence-freeze provenance drift")
    return provenance


def status(output: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "revision": REVISION,
        "output_root": rel(output),
    }
    for name in (
        PROTOCOL_NAME,
        INPUT_LEDGER_NAME,
        OUTPUT_LEDGER_NAME,
        "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json",
        "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json",
        SUPPORT_AUDIT_NAME,
        FREEZE_PROVENANCE_NAME,
        STATUS_NAME,
    ):
        path = output / name
        if path.is_file() and not path.is_symlink():
            value = strict_json(path) if path.suffix == ".json" else None
            result[name] = {
                "inventory": inventory(path),
                "status": value.get("status") if value is not None else None,
                "stage": value.get("stage") if value is not None else None,
            }
    return result


def _validate_output_root(output: Path) -> None:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise V1R3Error("v1r3 output root is fixed")
    if output.resolve() == PARENT_OUTPUT.resolve():
        raise V1R3Error("v1r3 cannot write into its immutable parent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "seal",
            "verify",
            "materialize",
            "audit",
            "freeze",
            "verify-freeze",
            "status",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    try:
        _validate_output_root(output)
        if args.command == "seal":
            value = seal(output)
        elif args.command == "verify":
            value = verify_protocol(output)
        elif args.command == "materialize":
            value = materialize(output)
        elif args.command == "audit":
            value = audit_complete_bundle(output)
        elif args.command == "freeze":
            value = freeze(output)
        elif args.command == "verify-freeze":
            value = verify_freeze(output)
        else:
            value = status(output)
        print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
        if args.command in {"freeze", "verify-freeze"} and value.get("status") == (
            "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL"
        ):
            return 3
        return 0
    except BaseException as error:
        try:
            if output == DEFAULT_OUTPUT.resolve():
                _write_status(output, {
                    "revision": REVISION,
                    "stage": "MF3ZU_V1R3_EVIDENCE_PROJECTION_TECHNICAL_FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "training_started": False,
                })
        except BaseException:
            pass
        print(
            f"MF3ZU_V1R3_PROJECTION_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
