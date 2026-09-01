#!/usr/bin/env python3
"""Outcome-blind MF3ZU v1r1 annotation contract recovery.

The parent 142/154 instruction result remains immutable.  This script widens
only the undisclosed instruction-list parser bound to the natural aNN range,
retries the 12 recorded failures once, builds an independent merged view, and
then runs the unchanged evidence question while retaining invalid responses.
It never opens ranking targets or model-performance artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Iterable, Mapping
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import annotate_mf3zu_rxr_evidence as parent  # noqa: E402
from revealnav_mf3.mf3zu_evidence_memory import (  # noqa: E402
    InstructionAtom,
    MF3ZUContractError,
    QWEN_ENABLE_THINKING,
    QWEN_ENDPOINT,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    REVISION as PARENT_REVISION,
    SEMANTIC_KIND_TO_EVIDENCE_TYPE,
    parse_instruction_response as parse_parent_instruction_response,
    reject_sensitive_mapping,
    stable_sha256,
    validate_evidence_response,
    validate_sha256,
)
from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    verify_protocol as verify_parent_protocol,
)


REVISION = "mf3zu_rxr_evidence_annotation_recovery_v1r1"
STATUS = "SEALED_BEFORE_MF3ZU_V1R1_ANNOTATION_RESPONSES"
ATOM_LIMIT = 99
INSTRUCTION_WORKERS = 4
EVIDENCE_WORKERS = 8
TRANSPORT_ATTEMPTS = 3
TRANSPORT_BACKOFF_SECONDS = 2.0

PARENT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_memory_feasibility_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_annotation_recovery_v1r1"
)
METHOD = ROOT / "METHOD_REVISION_3ZU_RXR_EVIDENCE_ANNOTATION_CONTRACT_REPAIR_V1R1.md"
RECOVERY_TEST = ROOT / "tests/test_annotation_recovery_mf3zu_v1r1.py"
PROTOCOL_NAME = "MF3ZU_V1R1_ANNOTATION_RECOVERY_PROTOCOL.json"
LEDGER_NAME = "MF3ZU_PARENT_ANNOTATION_TECHNICAL_FAILURE_LEDGER.json"
REPAIR_MANIFEST_NAME = "MF3ZU_V1R1_INSTRUCTION_REPAIR_MANIFEST.json"
MERGED_INDEX_NAME = "MF3ZU_V1R1_MERGED_INSTRUCTION_INDEX.jsonl"
STATUS_NAME = "MF3ZU_V1R1_ANNOTATION_RECOVERY_STATUS.json"

_WRITE_LOCK = threading.Lock()


class RecoveryError(RuntimeError):
    pass


_ATOM_ID = re.compile(r"^a[0-9]{2}$")


@dataclass(frozen=True)
class RecoveryInstructionGraph:
    """V1r1 graph with the natural positive sequential aNN support."""

    instruction_sha256: str
    atoms: tuple[InstructionAtom, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.instruction_sha256, field="instruction_sha256")
        if not self.atoms or len(self.atoms) > ATOM_LIMIT:
            raise MF3ZUContractError("invalid v1r1 instruction atom list")
        ids = [atom.instruction_atom_id for atom in self.atoms]
        if ids != [f"a{index:02d}" for index in range(1, len(ids) + 1)]:
            raise MF3ZUContractError("instruction atom IDs must be sequential")
        seen: set[str] = set()
        for atom in self.atoms:
            if any(value not in seen for value in atom.depends_on):
                raise MF3ZUContractError(
                    "dependencies must refer only to earlier atoms"
                )
            seen.add(atom.instruction_atom_id)

    def as_mapping(self) -> dict[str, object]:
        return {
            "instruction_sha256": self.instruction_sha256,
            "atoms": [atom.as_mapping() for atom in self.atoms],
        }


def parse_instruction_response_v1r1(
    value: object,
    *,
    instruction: str,
) -> RecoveryInstructionGraph:
    """Apply the parent atom semantics with only the 32 -> 99 bound fix."""

    if not isinstance(value, Mapping) or set(value) != {"instruction_atoms"}:
        raise MF3ZUContractError(
            "instruction response must contain only instruction_atoms"
        )
    rows = value["instruction_atoms"]
    if not isinstance(rows, list) or not rows or len(rows) > ATOM_LIMIT:
        raise MF3ZUContractError("invalid v1r1 instruction atom list")
    required = {
        "instruction_atom_id", "text", "semantic_kind", "depends_on"
    }
    atoms: list[InstructionAtom] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise MF3ZUContractError("instruction atom schema mismatch")
        dependencies = row["depends_on"]
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise MF3ZUContractError("instruction dependencies must be strings")
        kind = str(row["semantic_kind"])
        evidence_type = SEMANTIC_KIND_TO_EVIDENCE_TYPE.get(kind)
        if evidence_type is None:
            raise MF3ZUContractError("unknown instruction semantic kind")
        atom_id = str(row["instruction_atom_id"])
        if _ATOM_ID.fullmatch(atom_id) is None:
            raise MF3ZUContractError("instruction atom ID must match aNN")
        atoms.append(InstructionAtom(
            instruction_atom_id=atom_id,
            text=str(row["text"]).strip(),
            semantic_kind=kind,
            evidence_type=evidence_type,
            depends_on=tuple(dependencies),
        ))
    return RecoveryInstructionGraph(
        instruction_sha256=hashlib.sha256(
            instruction.strip().encode("utf-8")
        ).hexdigest(),
        atoms=tuple(atoms),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RecoveryError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RecoveryError(f"invalid regular file: {path}")
    return {
        "path": rel(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecoveryError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON object required: {path}")
    return value


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RecoveryError(f"cannot read JSONL: {path}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise RecoveryError(f"invalid JSONL: {path}:{number}") from error
        if not isinstance(value, dict):
            raise RecoveryError(f"JSONL object required: {path}:{number}")
        rows.append(value)
    if not rows:
        raise RecoveryError(f"empty JSONL: {path}")
    return rows


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RecoveryError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RecoveryError(f"stale partial output: {partial}")
    partial.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    refuse_existing: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RecoveryError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RecoveryError(f"stale partial output: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")
    os.replace(partial, path)


def atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RecoveryError(f"invalid copy source: {source}")
    payload = source.read_bytes()
    if destination.is_file() and not destination.is_symlink():
        if destination.read_bytes() != payload:
            raise RecoveryError(f"existing merged file differs: {destination}")
        return
    if destination.exists() or destination.is_symlink():
        raise RecoveryError(f"invalid copy destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RecoveryError(f"stale partial output: {partial}")
    partial.write_bytes(payload)
    os.replace(partial, destination)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        for row in rows
    ).encode("utf-8")


def _recoverable_write(path: Path, payload: bytes) -> None:
    """Finish or verify a deterministic two-file artifact commit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if path.is_file() and not path.is_symlink():
        if path.read_bytes() != payload:
            raise RecoveryError(f"existing deterministic artifact differs: {path}")
        if partial.exists() or partial.is_symlink():
            raise RecoveryError(f"orphan partial beside complete artifact: {partial}")
        return
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"invalid deterministic artifact path: {path}")
    if partial.is_file() and not partial.is_symlink():
        if partial.read_bytes() != payload:
            raise RecoveryError(f"stale partial artifact differs: {partial}")
        os.replace(partial, path)
        return
    if partial.exists() or partial.is_symlink():
        raise RecoveryError(f"invalid partial artifact: {partial}")
    partial.write_bytes(payload)
    os.replace(partial, path)


def recoverable_atomic_json(
    path: Path,
    value: object,
    *,
    refuse_existing: bool = False,
) -> None:
    del refuse_existing
    _recoverable_write(path, _json_bytes(value))


def recoverable_atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    refuse_existing: bool = False,
) -> None:
    del refuse_existing
    _recoverable_write(path, _jsonl_bytes(rows))


@contextmanager
def _exclusive_stage_lock(output: Path, stage: str):
    lock_path = output / ".locks" / f"{stage}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryError(f"recovery stage is already running: {stage}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def stage_locked(stage: str):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(output: Path, *args, **kwargs):
            with _exclusive_stage_lock(output, stage):
                return function(output, *args, **kwargs)
        return wrapped
    return decorate


def _request_intent_value(
    *,
    stage: str,
    request_id: str,
    identity: str,
    request_payload_sha256: str,
    protocol_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "revealnav-mf3zu-v1r1-request-intent/1",
        "revision": REVISION,
        "stage": stage,
        "request_id": request_id,
        "identity": identity,
        "request_payload_sha256": request_payload_sha256,
        "repair_protocol_sha256": protocol_sha256,
        "logical_attempt": 1,
        "candidate_target_accessed": False,
        "performance_accessed": False,
        "public_split_access": False,
    }


def _validate_request_intent(
    *,
    output: Path,
    stage: str,
    request_id: str,
    identity: str,
    request_payload_sha256: str,
    protocol_sha256: str,
) -> None:
    intent_path = output / "intents" / stage / f"{request_id}.json"
    expected = _request_intent_value(
        stage=stage,
        request_id=request_id,
        identity=identity,
        request_payload_sha256=request_payload_sha256,
        protocol_sha256=protocol_sha256,
    )
    if (
        not intent_path.is_file()
        or intent_path.is_symlink()
        or strict_json(intent_path) != expected
    ):
        raise RecoveryError(f"request intent provenance drift: {request_id}")


def _write_request_intent(
    *,
    output: Path,
    stage: str,
    request_id: str,
    identity: str,
    request_payload_sha256: str,
    protocol_sha256: str,
    response_path: Path,
) -> None:
    partial = response_path.with_name(response_path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RecoveryError(
            f"ambiguous response partial exists; refusing duplicate request: {request_id}"
        )
    intent_path = output / "intents" / stage / f"{request_id}.json"
    value = _request_intent_value(
        stage=stage,
        request_id=request_id,
        identity=identity,
        request_payload_sha256=request_payload_sha256,
        protocol_sha256=protocol_sha256,
    )
    if intent_path.is_file() and not intent_path.is_symlink():
        if strict_json(intent_path) != value:
            raise RecoveryError(f"request intent provenance drift: {request_id}")
        raise RecoveryError(
            f"ambiguous prior request intent; refusing duplicate request: {request_id}"
        )
    if intent_path.exists() or intent_path.is_symlink():
        raise RecoveryError(f"invalid request intent path: {intent_path}")
    atomic_json(intent_path, value, refuse_existing=True)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RecoveryError("cannot resolve source commit") from error


def _verify_implementation_at_commit(
    source_commit: str,
    records: Mapping[str, object],
) -> None:
    for name, raw in records.items():
        if not isinstance(raw, Mapping):
            raise RecoveryError(f"invalid implementation inventory: {name}")
        path = str(raw.get("path"))
        try:
            payload = subprocess.run(
                ["git", "show", f"{source_commit}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise RecoveryError(
                f"sealed implementation is absent from source commit: {path}"
            ) from error
        if (
            len(payload) != raw.get("bytes")
            or hashlib.sha256(payload).hexdigest() != raw.get("sha256")
        ):
            raise RecoveryError(
                f"sealed implementation differs from source commit: {path}"
            )


def _parent_requests() -> tuple[dict, list[dict]]:
    input_manifest = strict_json(
        PARENT_OUTPUT / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
    )
    if (
        input_manifest.get("revision") != PARENT_REVISION
        or input_manifest.get("status") != "SEALED_BEFORE_INSTRUCTION_RESPONSES"
        or input_manifest.get("episodes") != 154
        or input_manifest.get("model") != QWEN_MODEL
        or input_manifest.get("temperature") != QWEN_TEMPERATURE
        or input_manifest.get("max_tokens") != QWEN_MAX_TOKENS
        or input_manifest.get("thinking") is not QWEN_ENABLE_THINKING
        or input_manifest.get("ranking_label_read") is not False
        or input_manifest.get("task_metric_read") is not False
        or input_manifest.get("public_split_access") is not False
    ):
        raise RecoveryError("parent instruction inputs are not sealed")
    request_path = ROOT / str(input_manifest["requests"]["path"])
    if inventory(request_path) != input_manifest["requests"]:
        raise RecoveryError("parent instruction requests changed")
    requests = jsonl(request_path)
    if (
        len(requests) != 154
        or len({str(row.get("request_id")) for row in requests}) != 154
        or len({str(row.get("episode_id")) for row in requests}) != 154
    ):
        raise RecoveryError("parent instruction request identity drift")
    return input_manifest, requests


def classify_parent() -> dict[str, object]:
    verify_parent_protocol()
    input_manifest, requests = _parent_requests()
    annotation_path = PARENT_OUTPUT / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
    annotation = strict_json(annotation_path)
    if (
        annotation.get("revision") != PARENT_REVISION
        or annotation.get("status") != "FAIL"
        or annotation.get("planned") != 154
        or annotation.get("response_files") != 154
        or annotation.get("pass") != 142
        or annotation.get("model") != QWEN_MODEL
        or annotation.get("ranking_label_read") is not False
        or annotation.get("task_metric_read") is not False
        or annotation.get("public_split_access") is not False
    ):
        raise RecoveryError("parent technical-failure manifest drift")
    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    response_bundle: list[dict[str, object]] = []
    for row in requests:
        request_id = str(row["request_id"])
        path = PARENT_OUTPUT / "responses/instruction" / f"{request_id}.json"
        response = strict_json(path)
        response_bundle.append(inventory(path))
        if (
            response.get("request_id") != request_id
            or response.get("ranking_label_read") is not False
            or response.get("task_metric_read") is not False
            or response.get("public_split_access") is not False
        ):
            raise RecoveryError("parent instruction response provenance drift")
        if response.get("status") == "PASS":
            if (
                response.get("model_requested") != QWEN_MODEL
                or response.get("provider_model") != QWEN_MODEL
                or response.get("temperature") != QWEN_TEMPERATURE
                or response.get("max_tokens") != QWEN_MAX_TOKENS
                or response.get("thinking") is not QWEN_ENABLE_THINKING
            ):
                raise RecoveryError("parent PASS model provenance drift")
            parse_parent_instruction_response(
                response.get("response"),
                instruction=str(row["instruction"]),
            )
            passed.append({
                "episode_id": str(row["episode_id"]),
                "request_id": request_id,
                "response": inventory(path),
            })
        else:
            if response.get("error") != (
                "MF3ZUContractError: invalid instruction atom list"
            ):
                raise RecoveryError("parent failure class changed")
            failed.append({
                "episode_id": str(row["episode_id"]),
                "request_id": request_id,
                "instruction_sha256": hashlib.sha256(
                    str(row["instruction"]).strip().encode("utf-8")
                ).hexdigest(),
                "instruction_characters": len(str(row["instruction"])),
                "error": str(response["error"]),
                "response": inventory(path),
            })
    passed.sort(key=lambda row: str(row["request_id"]))
    failed.sort(key=lambda row: str(row["request_id"]))
    response_bundle.sort(key=lambda row: str(row["path"]))
    manifest_ids = sorted(
        str(row.get("request_id")) for row in annotation.get("failures", [])
    )
    if (
        len(passed) != 142
        or len(failed) != 12
        or manifest_ids != [str(row["request_id"]) for row in failed]
    ):
        raise RecoveryError("parent pass/failure inventory drift")
    return {
        "input_manifest": input_manifest,
        "requests": requests,
        "annotation": annotation,
        "passed": passed,
        "failed": failed,
        "response_bundle": response_bundle,
        "response_bundle_sha256": stable_sha256(response_bundle),
    }


def _ledger_value(state: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "revealnav-mf3zu-parent-annotation-failure-ledger/1",
        "revision": REVISION,
        "status": "PARENT_TECHNICAL_FAILURE_IMMUTABLY_RECORDED",
        "parent_revision": PARENT_REVISION,
        "parent_instruction_annotation": inventory(
            PARENT_OUTPUT / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
        ),
        "parent_supervisor_status": inventory(
            PARENT_OUTPUT / "MF3ZU_RXR_PIPELINE_STATUS.json"
        ),
        "parent_supervisor_log": inventory(
            PARENT_OUTPUT / "MF3ZU_RXR_PIPELINE_SUPERVISOR.log"
        ),
        "current_state": {
            "planned": 154,
            "pass": 142,
            "fail": 12,
            "failure_class": "MF3ZUContractError: invalid instruction atom list",
            "failed_requests": list(state["failed"]),
            "response_bundle_sha256": state["response_bundle_sha256"],
            "evidence_prepared": False,
            "evidence_annotated": False,
            "model_trained": False,
            "performance_read": False,
        },
        "execution_history_note": (
            "supervisor log records an earlier 141/154 state; identical retries "
            "were later overwritten by the parent runner, leaving the current "
            "immutable response bundle at 142/154"
        ),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    }


def _protocol_value(
    state: Mapping[str, object],
    *,
    source_commit: str,
    output: Path,
) -> dict[str, object]:
    ledger_path = output / LEDGER_NAME
    return {
        "schema_version": "revealnav-mf3zu-annotation-recovery-protocol/1",
        "revision": REVISION,
        "status": STATUS,
        "source_commit": source_commit,
        "implementation": {
            "method": inventory(METHOD),
            "runner": inventory(Path(__file__).resolve()),
            "regression_test": inventory(RECOVERY_TEST),
        },
        "parent": {
            "protocol": inventory(
                PARENT_OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PROTOCOL.json"
            ),
            "population_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json"
            ),
            "observation_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json"
            ),
            "instruction_input_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
            ),
            "instruction_requests": dict(state["input_manifest"])["requests"],
            "instruction_annotation_manifest": inventory(
                PARENT_OUTPUT / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
            ),
            "response_bundle_sha256": state["response_bundle_sha256"],
            "technical_failure_ledger": inventory(ledger_path),
            "parent_files_read_only": True,
        },
        "instruction_repair": {
            "selection": "exact 12 failed request IDs in parent manifest",
            "failed_requests": list(state["failed"]),
            "existing_pass_responses": 142,
            "existing_pass_responses_read_only": True,
            "parser_atom_limit_before": 32,
            "parser_atom_limit_after": ATOM_LIMIT,
            "bound_source": "positive sequential two-digit aNN IDs: a01..a99",
            "semantic_schema_change": False,
            "prompt_change": False,
            "logical_attempts_per_failed_request": 1,
            "invalid_parsed_response_retained": True,
            "historical_source_files_modified": False,
            "graph_loader": "v1r1 runtime adapter in sealed runner",
        },
        "evidence_annotation": {
            "decisions": 1428,
            "episodes": 154,
            "raw_scenes": 59,
            "prompt_change": False,
            "validator_change": False,
            "logical_attempts_per_decision": 1,
            "invalid_parsed_response_retained": True,
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "thinking": QWEN_ENABLE_THINKING,
            "max_tokens": QWEN_MAX_TOKENS,
        },
        "transport": {
            "attempts": TRANSPORT_ATTEMPTS,
            "backoff_seconds": TRANSPORT_BACKOFF_SECONDS,
            "instruction_workers": INSTRUCTION_WORKERS,
            "evidence_workers": EVIDENCE_WORKERS,
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
            "training_authorized_by_this_runner": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
        },
    }


@stage_locked("seal")
def seal(output: Path) -> dict:
    protocol_path = output / PROTOCOL_NAME
    if protocol_path.is_file() and not protocol_path.is_symlink():
        return verify_protocol(output)
    if protocol_path.exists() or protocol_path.is_symlink():
        raise RecoveryError(f"invalid recovery protocol path: {protocol_path}")
    state = classify_parent()
    source_commit = _git_head()
    implementation = {
        "method": inventory(METHOD),
        "runner": inventory(Path(__file__).resolve()),
        "regression_test": inventory(RECOVERY_TEST),
    }
    _verify_implementation_at_commit(source_commit, implementation)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / LEDGER_NAME
    recoverable_atomic_json(ledger_path, _ledger_value(state))
    value = _protocol_value(
        state,
        source_commit=source_commit,
        output=output,
    )
    recoverable_atomic_json(protocol_path, value)
    return value


def verify_protocol(output: Path) -> dict:
    protocol_path = output / PROTOCOL_NAME
    value = strict_json(protocol_path)
    if value.get("revision") != REVISION or value.get("status") != STATUS:
        raise RecoveryError("recovery protocol identity drift")
    state = classify_parent()
    ledger = strict_json(output / LEDGER_NAME)
    if ledger != _ledger_value(state):
        raise RecoveryError("parent failure ledger drift")
    expected = _protocol_value(
        state,
        source_commit=str(value.get("source_commit")),
        output=output,
    )
    if value != expected:
        raise RecoveryError("recovery protocol/source drift")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{value['source_commit']}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise RecoveryError("sealed source commit is unavailable") from error
    implementation = value.get("implementation")
    if not isinstance(implementation, Mapping):
        raise RecoveryError("recovery implementation inventory is malformed")
    _verify_implementation_at_commit(str(value["source_commit"]), implementation)
    return value


def _api_key() -> str:
    return parent._api_key()


def _provider_request_preserving(
    api_key: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    reject_sensitive_mapping(payload)
    request = urllib.request.Request(
        QWEN_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    attempts: list[dict[str, object]] = []
    last_raw: str | None = None
    for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
        body_text: str | None = None
        raw: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body_text = response.read().decode("utf-8")
            body = json.loads(body_text)
            message = body["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(
                    str(item.get("text", ""))
                    if isinstance(item, Mapping) else str(item)
                    for item in message
                )
            raw = str(message)
            last_raw = raw
            parsed = json.loads(raw)
            attempts.append({
                "attempt": attempt,
                "status": "PARSED_JSON",
                "provider_model": body.get("model"),
                "raw_content_sha256": hashlib.sha256(
                    raw.encode("utf-8")
                ).hexdigest(),
            })
            return {
                "status": "PARSED_JSON",
                "provider_model": body.get("model"),
                "response": parsed,
                "raw_content": raw,
                "usage": body.get("usage"),
                "transport_attempts": attempts,
            }
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            urllib.error.HTTPError,
        ) as error:
            failed_attempt: dict[str, object] = {
                "attempt": attempt,
                "status": "TRANSPORT_OR_JSON_FAIL",
                "error": type(error).__name__,
            }
            if raw is not None:
                failed_attempt.update({
                    "raw_provider_content": raw,
                    "raw_content_sha256": hashlib.sha256(
                        raw.encode("utf-8")
                    ).hexdigest(),
                })
            elif body_text is not None:
                failed_attempt.update({
                    "raw_http_body": body_text,
                    "raw_http_body_sha256": hashlib.sha256(
                        body_text.encode("utf-8")
                    ).hexdigest(),
                })
            attempts.append(failed_attempt)
            if attempt < TRANSPORT_ATTEMPTS:
                time.sleep(TRANSPORT_BACKOFF_SECONDS)
    return {
        "status": "FAIL",
        "error": str(attempts[-1]["error"] if attempts else "unknown"),
        "raw_content": last_raw,
        "transport_attempts": attempts,
    }


def _validate_fixed_payload(payload: Mapping[str, object]) -> None:
    if (
        payload.get("model") != QWEN_MODEL
        or payload.get("temperature") != QWEN_TEMPERATURE
        or payload.get("max_tokens") != QWEN_MAX_TOKENS
        or payload.get("enable_thinking") is not QWEN_ENABLE_THINKING
    ):
        raise RecoveryError("fixed Qwen payload drift")
    reject_sensitive_mapping(payload)


def _repair_instruction_one(
    *,
    api_key: str,
    row: Mapping[str, object],
    output: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    request_id = str(row["request_id"])
    destination = output / "responses/instruction_repair" / f"{request_id}.json"
    if destination.exists() or destination.is_symlink():
        value = strict_json(destination)
        _validate_instruction_repair_record(
            row,
            value,
            output=output,
            protocol_sha256=protocol_sha256,
        )
        return value
    payload = dict(row["payload"])
    _validate_fixed_payload(payload)
    payload_sha256 = stable_sha256(payload)
    _write_request_intent(
        output=output,
        stage="instruction_repair",
        request_id=request_id,
        identity=str(row["episode_id"]),
        request_payload_sha256=payload_sha256,
        protocol_sha256=protocol_sha256,
        response_path=destination,
    )
    provider = _provider_request_preserving(api_key, payload)
    error: str | None = None
    parsed = provider.get("response")
    if provider.get("status") == "PARSED_JSON":
        if provider.get("provider_model") != QWEN_MODEL:
            error = "RecoveryError: provider model identity drift"
        else:
            try:
                parse_instruction_response_v1r1(
                    parsed,
                    instruction=str(row["instruction"]),
                )
            except BaseException as caught:
                error = f"{type(caught).__name__}: {caught}"
    else:
        error = f"RecoveryTransportError: {provider.get('error')}"
    passed = error is None
    value: dict[str, object] = {
        "schema_version": "revealnav-mf3zu-instruction-recovery-response/1",
        "revision": REVISION,
        "status": "PASS" if passed else "FAIL",
        "stage": "instruction_repair",
        "request_id": request_id,
        "episode_id": str(row["episode_id"]),
        "model_requested": QWEN_MODEL,
        "provider_model": provider.get("provider_model"),
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "thinking": QWEN_ENABLE_THINKING,
        "instruction_atom_limit": ATOM_LIMIT,
        "request_payload_sha256": payload_sha256,
        "repair_protocol_sha256": protocol_sha256,
        "transport_attempts": provider.get("transport_attempts"),
        "usage": provider.get("usage"),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }
    if passed:
        value["response"] = parsed
    else:
        value["error"] = error
        value["raw_provider_content"] = provider.get("raw_content")
        if parsed is not None:
            value["invalid_parsed_response"] = parsed
    atomic_json(destination, value, refuse_existing=True)
    return value


def _validate_instruction_repair_record(
    row: Mapping[str, object],
    value: Mapping[str, object],
    *,
    output: Path,
    protocol_sha256: str,
) -> None:
    _validate_request_intent(
        output=output,
        stage="instruction_repair",
        request_id=str(row["request_id"]),
        identity=str(row["episode_id"]),
        request_payload_sha256=stable_sha256(row["payload"]),
        protocol_sha256=protocol_sha256,
    )
    if (
        value.get("revision") != REVISION
        or value.get("request_id") != row.get("request_id")
        or value.get("episode_id") != str(row.get("episode_id"))
        or value.get("model_requested") != QWEN_MODEL
        or value.get("instruction_atom_limit") != ATOM_LIMIT
        or value.get("request_payload_sha256") != stable_sha256(row["payload"])
        or value.get("repair_protocol_sha256") != protocol_sha256
        or value.get("ranking_label_read") is not False
        or value.get("task_metric_read") is not False
        or value.get("public_split_access") is not False
    ):
        raise RecoveryError("instruction repair response provenance drift")
    if value.get("status") == "PASS":
        if value.get("provider_model") != QWEN_MODEL:
            raise RecoveryError("instruction repair provider identity drift")
        parse_instruction_response_v1r1(
            value.get("response"),
            instruction=str(row["instruction"]),
        )
    elif value.get("status") != "FAIL":
        raise RecoveryError("instruction repair status is invalid")


@stage_locked("instruction_annotation")
def annotate_instructions(output: Path) -> dict:
    protocol = verify_protocol(output)
    state = classify_parent()
    by_id = {str(row["request_id"]): row for row in state["requests"]}
    failed_ids = [str(row["request_id"]) for row in state["failed"]]
    protocol_sha256 = sha256_file(output / PROTOCOL_NAME)
    api_key = _api_key()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=INSTRUCTION_WORKERS) as pool:
        futures = [
            pool.submit(
                _repair_instruction_one,
                api_key=api_key,
                row=by_id[request_id],
                output=output,
                protocol_sha256=protocol_sha256,
            )
            for request_id in failed_ids
        ]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: str(row["request_id"]))
    passed = sum(row.get("status") == "PASS" for row in results)
    manifest = {
        "schema_version": "revealnav-mf3zu-instruction-recovery/1",
        "revision": REVISION,
        "status": "PASS" if passed == 12 else "FAIL",
        "planned": 12,
        "pass": passed,
        "fail": 12 - passed,
        "instruction_atom_limit": ATOM_LIMIT,
        "parent_pass_responses": 142,
        "merged_coverage_if_pass": 142 + passed,
        "failed_request_ids": failed_ids,
        "failures": [
            {
                "request_id": row["request_id"],
                "error": row.get("error"),
            }
            for row in results if row.get("status") != "PASS"
        ],
        "response_bundle_sha256": stable_sha256([
            inventory(
                output / "responses/instruction_repair"
                / f"{row['request_id']}.json"
            )
            for row in results
        ]),
        "source_protocol": inventory(output / PROTOCOL_NAME),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }
    recoverable_atomic_json(output / REPAIR_MANIFEST_NAME, manifest)
    return manifest


def _assert_no_target_named_file(output: Path) -> None:
    offenders = [
        rel(path) for path in output.rglob("*")
        if "target" in path.name.casefold()
    ]
    if offenders:
        raise RecoveryError(f"target-named file entered recovery view: {offenders[:2]}")


@stage_locked("merged_view")
def materialize_merged_view(output: Path) -> dict:
    verify_protocol(output)
    repair_manifest = strict_json(output / REPAIR_MANIFEST_NAME)
    if (
        repair_manifest.get("revision") != REVISION
        or repair_manifest.get("status") != "PASS"
        or repair_manifest.get("planned") != 12
        or repair_manifest.get("pass") != 12
        or repair_manifest.get("instruction_atom_limit") != ATOM_LIMIT
    ):
        raise RecoveryError("instruction repair is incomplete")
    state = classify_parent()
    requests = list(state["requests"])
    failed_ids = {str(row["request_id"]) for row in state["failed"]}
    for name in (
        "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json",
        "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json",
        "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json",
    ):
        atomic_copy(PARENT_OUTPUT / name, output / name)
    index_rows: list[dict[str, object]] = []
    for row in requests:
        request_id = str(row["request_id"])
        if request_id in failed_ids:
            source = (
                output / "responses/instruction_repair" / f"{request_id}.json"
            )
            source_kind = "v1r1_repair"
        else:
            source = (
                PARENT_OUTPUT / "responses/instruction" / f"{request_id}.json"
            )
            source_kind = "parent_pass"
        response = strict_json(source)
        if response.get("status") != "PASS":
            raise RecoveryError("non-passing response entered merged view")
        if request_id in failed_ids:
            _validate_instruction_repair_record(
                row,
                response,
                output=output,
                protocol_sha256=sha256_file(output / PROTOCOL_NAME),
            )
        graph = parse_instruction_response_v1r1(
            response.get("response"),
            instruction=str(row["instruction"]),
        )
        destination = output / "responses/instruction" / f"{request_id}.json"
        atomic_copy(source, destination)
        index_rows.append({
            "episode_id": str(row["episode_id"]),
            "request_id": request_id,
            "source_kind": source_kind,
            "source_response": inventory(source),
            "merged_response": inventory(destination),
            "instruction_atom_count": len(graph.atoms),
            "instruction_graph_sha256": stable_sha256(graph.as_mapping()),
        })
    index_rows.sort(key=lambda row: str(row["request_id"]))
    if (
        len(index_rows) != 154
        or len({str(row["episode_id"]) for row in index_rows}) != 154
        or sum(row["source_kind"] == "parent_pass" for row in index_rows) != 142
        or sum(row["source_kind"] == "v1r1_repair" for row in index_rows) != 12
    ):
        raise RecoveryError("merged instruction coverage drift")
    index_path = output / MERGED_INDEX_NAME
    if not index_path.exists():
        atomic_jsonl(index_path, index_rows, refuse_existing=True)
    elif jsonl(index_path) != index_rows:
        raise RecoveryError("merged instruction index changed")
    manifest = {
        "schema_version": "revealnav-mf3zu-instruction-annotation-merged/1",
        # Parent evidence/memory utilities deliberately retain their original
        # scientific revision identity; the correction provenance is explicit.
        "revision": PARENT_REVISION,
        "contract_repair_revision": REVISION,
        "status": "PASS",
        "planned": 154,
        "response_files": 154,
        "pass": 154,
        "fail": 0,
        "model": QWEN_MODEL,
        "instruction_atom_limit": ATOM_LIMIT,
        "merged_index": inventory(index_path),
        "parent_annotation_manifest": inventory(
            PARENT_OUTPUT / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
        ),
        "repair_manifest": inventory(output / REPAIR_MANIFEST_NAME),
        "repair_protocol": inventory(output / PROTOCOL_NAME),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }
    destination = output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
    if destination.exists():
        if strict_json(destination) != manifest:
            raise RecoveryError("merged instruction manifest changed")
    else:
        atomic_json(destination, manifest, refuse_existing=True)
    _assert_no_target_named_file(output)
    return manifest


def _recovery_instruction_graphs(
    output: Path,
) -> tuple[dict[str, RecoveryInstructionGraph], dict[str, str]]:
    annotation = strict_json(
        output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
    )
    if (
        annotation.get("revision") != PARENT_REVISION
        or annotation.get("contract_repair_revision") != REVISION
        or annotation.get("status") != "PASS"
        or annotation.get("pass") != 154
        or annotation.get("instruction_atom_limit") != ATOM_LIMIT
    ):
        raise RecoveryError("v1r1 instruction annotation is incomplete")
    request_manifest = strict_json(
        output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
    )
    request_path = ROOT / str(request_manifest["requests"]["path"])
    if inventory(request_path) != request_manifest["requests"]:
        raise RecoveryError("sealed instruction request source changed")
    index_info = annotation.get("merged_index")
    if not isinstance(index_info, Mapping):
        raise RecoveryError("merged instruction index inventory is missing")
    index_path = ROOT / str(index_info.get("path"))
    if inventory(index_path) != dict(index_info):
        raise RecoveryError("merged instruction index changed")
    index_rows = jsonl(index_path)
    index_by_request = {
        str(row.get("request_id")): row for row in index_rows
    }
    if len(index_rows) != 154 or len(index_by_request) != 154:
        raise RecoveryError("merged instruction index identity drift")
    graphs: dict[str, RecoveryInstructionGraph] = {}
    instructions: dict[str, str] = {}
    for row in jsonl(request_path):
        request_id = str(row["request_id"])
        response_path = output / "responses/instruction" / f"{request_id}.json"
        index_row = index_by_request.get(request_id)
        if not isinstance(index_row, Mapping):
            raise RecoveryError("instruction response is absent from merged index")
        if inventory(response_path) != index_row.get("merged_response"):
            raise RecoveryError("merged instruction response inventory drift")
        source_info = index_row.get("source_response")
        if not isinstance(source_info, Mapping):
            raise RecoveryError("merged instruction source inventory is missing")
        if inventory(ROOT / str(source_info.get("path"))) != dict(source_info):
            raise RecoveryError("merged instruction source response changed")
        response = strict_json(response_path)
        if response.get("status") != "PASS":
            raise RecoveryError("non-passing merged instruction response")
        graph = parse_instruction_response_v1r1(
            response.get("response"),
            instruction=str(row["instruction"]),
        )
        if (
            index_row.get("episode_id") != str(row["episode_id"])
            or index_row.get("instruction_atom_count") != len(graph.atoms)
            or index_row.get("instruction_graph_sha256")
            != stable_sha256(graph.as_mapping())
        ):
            raise RecoveryError("merged instruction graph provenance drift")
        episode_id = str(row["episode_id"])
        if episode_id in graphs:
            raise RecoveryError("duplicate merged instruction episode")
        graphs[episode_id] = graph
        instructions[episode_id] = str(row["instruction"])
    if len(graphs) != 154:
        raise RecoveryError("merged instruction graph coverage drift")
    return graphs, instructions


@stage_locked("evidence_prepare")
def prepare_evidence(output: Path) -> dict:
    verify_protocol(output)
    materialize_merged_view(output)
    destination = output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
    if destination.exists() or destination.is_symlink():
        value = strict_json(destination)
        if (
            value.get("status") != "SEALED_BEFORE_EVIDENCE_RESPONSES"
            or value.get("request_count") != 1428
        ):
            raise RecoveryError("existing evidence input is invalid")
        return value
    original_loader = parent._instruction_graphs
    original_atomic_json = parent.atomic_json
    original_atomic_jsonl = parent.atomic_jsonl
    parent._instruction_graphs = _recovery_instruction_graphs
    parent.atomic_json = recoverable_atomic_json
    parent.atomic_jsonl = recoverable_atomic_jsonl
    try:
        value = parent.prepare_evidence(output)
    finally:
        parent._instruction_graphs = original_loader
        parent.atomic_json = original_atomic_json
        parent.atomic_jsonl = original_atomic_jsonl
    _assert_no_target_named_file(output)
    return value


def _validate_evidence_record(
    row: Mapping[str, object],
    value: Mapping[str, object],
    *,
    graph: object,
    output: Path,
    protocol_sha256: str,
    request_payload_sha256: str,
) -> None:
    _validate_request_intent(
        output=output,
        stage="evidence",
        request_id=str(row["request_id"]),
        identity=str(row["event_id"]),
        request_payload_sha256=request_payload_sha256,
        protocol_sha256=protocol_sha256,
    )
    if (
        value.get("revision") != PARENT_REVISION
        or value.get("contract_repair_revision") != REVISION
        or value.get("request_id") != row.get("request_id")
        or value.get("event_id") != str(row.get("event_id"))
        or value.get("model_requested") != QWEN_MODEL
        or value.get("temperature") != QWEN_TEMPERATURE
        or value.get("max_tokens") != QWEN_MAX_TOKENS
        or value.get("thinking") is not QWEN_ENABLE_THINKING
        or value.get("request_payload_sha256") != request_payload_sha256
        or value.get("repair_protocol_sha256") != protocol_sha256
        or value.get("ranking_label_read") is not False
        or value.get("task_metric_read") is not False
        or value.get("public_split_access") is not False
        or value.get("human_verified") is not False
        or value.get("gold") is not False
    ):
        raise RecoveryError("evidence response provenance drift")
    if value.get("status") == "PASS":
        if value.get("provider_model") != QWEN_MODEL:
            raise RecoveryError("evidence provider identity drift")
        validate_evidence_response(
            value.get("response"),
            graph=graph,
            decision_step=int(row["decision_step"]),
            allowed_candidate_ids=list(row["candidate_alias_to_action_id"]),
        )
    elif value.get("status") != "FAIL" or not value.get("error"):
        raise RecoveryError("evidence response status is invalid")


def _annotate_evidence_one(
    *,
    api_key: str,
    row: Mapping[str, object],
    graph: object,
    output: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    request_id = str(row["request_id"])
    destination = output / "responses/evidence" / f"{request_id}.json"
    payload = parent._evidence_payload(row)
    _validate_fixed_payload(payload)
    payload_sha256 = stable_sha256(payload)
    if destination.exists() or destination.is_symlink():
        value = strict_json(destination)
        _validate_evidence_record(
            row,
            value,
            graph=graph,
            output=output,
            protocol_sha256=protocol_sha256,
            request_payload_sha256=payload_sha256,
        )
        return value
    _write_request_intent(
        output=output,
        stage="evidence",
        request_id=request_id,
        identity=str(row["event_id"]),
        request_payload_sha256=payload_sha256,
        protocol_sha256=protocol_sha256,
        response_path=destination,
    )
    provider = _provider_request_preserving(api_key, payload)
    parsed = provider.get("response")
    error: str | None = None
    if provider.get("status") == "PARSED_JSON":
        if provider.get("provider_model") != QWEN_MODEL:
            error = "RecoveryError: provider model identity drift"
        else:
            try:
                validate_evidence_response(
                    parsed,
                    graph=graph,
                    decision_step=int(row["decision_step"]),
                    allowed_candidate_ids=list(row["candidate_alias_to_action_id"]),
                )
            except BaseException as caught:
                error = f"{type(caught).__name__}: {caught}"
    else:
        error = f"RecoveryTransportError: {provider.get('error')}"
    passed = error is None
    value: dict[str, object] = {
        "schema_version": "revealnav-mf3zu-evidence-recovery-response/1",
        "revision": PARENT_REVISION,
        "contract_repair_revision": REVISION,
        "status": "PASS" if passed else "FAIL",
        "stage": "evidence",
        "request_id": request_id,
        "event_id": str(row["event_id"]),
        "model_requested": QWEN_MODEL,
        "provider_model": provider.get("provider_model"),
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "thinking": QWEN_ENABLE_THINKING,
        "request_payload_sha256": payload_sha256,
        "repair_protocol_sha256": protocol_sha256,
        "transport_attempts": provider.get("transport_attempts"),
        "usage": provider.get("usage"),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }
    if passed:
        value["response"] = parsed
    else:
        value["error"] = error
        value["raw_provider_content"] = provider.get("raw_content")
        if parsed is not None:
            value["invalid_parsed_response"] = parsed
    atomic_json(destination, value, refuse_existing=True)
    return value


def _evidence_rows_and_graphs(output: Path) -> tuple[list[dict], dict[str, object]]:
    manifest = strict_json(output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json")
    if (
        manifest.get("revision") != PARENT_REVISION
        or manifest.get("status") != "SEALED_BEFORE_EVIDENCE_RESPONSES"
        or manifest.get("request_count") != 1428
        or manifest.get("episodes") != 154
        or manifest.get("ranking_label_read") is not False
        or manifest.get("task_metric_read") is not False
        or manifest.get("public_split_access") is not False
    ):
        raise RecoveryError("evidence inputs are not sealed")
    request_path = ROOT / str(manifest["requests"]["path"])
    if inventory(request_path) != manifest["requests"]:
        raise RecoveryError("evidence request artifact changed")
    rows = jsonl(request_path)
    graphs, _ = _recovery_instruction_graphs(output)
    if len(rows) != 1428 or len(graphs) != 154:
        raise RecoveryError("evidence request/graph cardinality drift")
    return rows, graphs


def audit_complete_evidence_bundle(output: Path) -> dict[str, object]:
    """Re-authenticate every response immediately before memory materialization."""

    manifest_path = output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("revision") != PARENT_REVISION
        or manifest.get("contract_repair_revision") != REVISION
        or manifest.get("status") != "PASS"
        or manifest.get("planned") != 1428
        or manifest.get("response_files") != 1428
        or manifest.get("pass") != 1428
        or manifest.get("fail") != 0
        or manifest.get("failures") != []
        or manifest.get("model") != QWEN_MODEL
        or manifest.get("ranking_label_read") is not False
        or manifest.get("task_metric_read") is not False
        or manifest.get("public_split_access") is not False
        or manifest.get("human_verified") is not False
        or manifest.get("gold") is not False
    ):
        raise RecoveryError("complete evidence manifest contract drift")
    input_path = output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
    if manifest.get("source_evidence_input_manifest") != inventory(input_path):
        raise RecoveryError("evidence manifest input provenance drift")
    if manifest.get("repair_protocol") != inventory(output / PROTOCOL_NAME):
        raise RecoveryError("evidence manifest protocol provenance drift")
    rows, graphs = _evidence_rows_and_graphs(output)
    protocol_sha256 = sha256_file(output / PROTOCOL_NAME)
    inventories: list[dict[str, object]] = []
    for row in rows:
        payload = parent._evidence_payload(row)
        _validate_fixed_payload(payload)
        response_path = (
            output / "responses/evidence" / f"{row['request_id']}.json"
        )
        response = strict_json(response_path)
        _validate_evidence_record(
            row,
            response,
            graph=graphs[str(row["episode_id"])],
            output=output,
            protocol_sha256=protocol_sha256,
            request_payload_sha256=stable_sha256(payload),
        )
        if response.get("status") != "PASS":
            raise RecoveryError("non-passing evidence response entered bundle")
        inventories.append(inventory(response_path))
    bundle_sha256 = stable_sha256(inventories)
    if manifest.get("response_bundle_sha256") != bundle_sha256:
        raise RecoveryError("evidence response bundle SHA drift")
    return {
        "responses": len(inventories),
        "response_bundle_sha256": bundle_sha256,
        "manifest": inventory(manifest_path),
    }


def _write_status(output: Path, value: Mapping[str, object]) -> None:
    with _WRITE_LOCK:
        path = output / STATUS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        partial.write_bytes(_json_bytes(dict(value)))
        os.replace(partial, path)


@stage_locked("evidence_annotation")
def annotate_evidence(output: Path) -> dict:
    verify_protocol(output)
    rows, graphs = _evidence_rows_and_graphs(output)
    protocol_sha256 = sha256_file(output / PROTOCOL_NAME)
    api_key = _api_key()
    completed = 0
    completed_lock = threading.Lock()

    def one(row: dict) -> dict[str, object]:
        nonlocal completed
        value = _annotate_evidence_one(
            api_key=api_key,
            row=row,
            graph=graphs[str(row["episode_id"])],
            output=output,
            protocol_sha256=protocol_sha256,
        )
        with completed_lock:
            completed += 1
            current = completed
        if current == 1 or current % 25 == 0 or current == len(rows):
            _write_status(output, {
                "revision": REVISION,
                "stage": "EVIDENCE_ANNOTATION_RUNNING",
                "completed_this_invocation": current,
                "planned": len(rows),
            })
        return value

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=EVIDENCE_WORKERS) as pool:
        futures = [pool.submit(one, row) for row in rows]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda value: str(value["request_id"]))
    passed = sum(value.get("status") == "PASS" for value in results)
    failures = [
        {"request_id": value["request_id"], "error": value.get("error")}
        for value in results if value.get("status") != "PASS"
    ]
    response_inventories = [
        inventory(output / "responses/evidence" / f"{row['request_id']}.json")
        for row in rows
    ]
    manifest = {
        "schema_version": "revealnav-mf3zu-evidence-annotation/1",
        "revision": PARENT_REVISION,
        "contract_repair_revision": REVISION,
        "status": "PASS" if passed == 1428 else "FAIL",
        "planned": 1428,
        "response_files": len(response_inventories),
        "pass": passed,
        "fail": 1428 - passed,
        "failures": failures,
        "model": QWEN_MODEL,
        "response_bundle_sha256": stable_sha256(response_inventories),
        "source_evidence_input_manifest": inventory(
            output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
        ),
        "repair_protocol": inventory(output / PROTOCOL_NAME),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "human_verified": False,
        "gold": False,
    }
    recoverable_atomic_json(
        output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json", manifest
    )
    _write_status(output, {
        "revision": REVISION,
        "stage": "EVIDENCE_ANNOTATION_COMPLETE" if passed == 1428 else "EVIDENCE_ANNOTATION_TECHNICAL_FAIL",
        "planned": 1428,
        "pass": passed,
        "fail": 1428 - passed,
    })
    return manifest


@stage_locked("evidence_freeze")
def freeze_evidence(output: Path) -> dict:
    verify_protocol(output)
    bundle_audit = audit_complete_evidence_bundle(output)
    # Imported only after complete outcome-blind evidence annotation.  The
    # builder materializes deterministic memories and does not open targets.
    import build_mf3zu_evidence_memory as builder
    if builder.annotation is not parent:
        raise RecoveryError("memory builder annotation module identity drift")
    original_loader = parent._instruction_graphs
    original_atomic_json = builder.atomic_json
    original_atomic_jsonl = builder.atomic_jsonl
    parent._instruction_graphs = _recovery_instruction_graphs
    builder.atomic_json = recoverable_atomic_json
    builder.atomic_jsonl = recoverable_atomic_jsonl
    try:
        value = builder.build(output)
    finally:
        parent._instruction_graphs = original_loader
        builder.atomic_json = original_atomic_json
        builder.atomic_jsonl = original_atomic_jsonl
    provenance = {
        "schema_version": "revealnav-mf3zu-v1r1-evidence-freeze/1",
        "revision": REVISION,
        "status": value["status"],
        "source_protocol": inventory(output / PROTOCOL_NAME),
        "source_instruction_manifest": inventory(
            output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
        ),
        "source_evidence_manifest": inventory(
            output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
        ),
        "response_bundle_audit": bundle_audit,
        "evidence_memory_manifest": inventory(
            output / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json"
        ),
        "candidate_target_accessed": False,
        "performance_accessed": False,
        "public_split_access": False,
        "training_run": False,
    }
    recoverable_atomic_json(
        output / "MF3ZU_V1R1_EVIDENCE_FREEZE_PROVENANCE.json",
        provenance,
    )
    return provenance


def status(output: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "revision": REVISION,
        "output_root": rel(output),
    }
    for name in (
        LEDGER_NAME,
        PROTOCOL_NAME,
        REPAIR_MANIFEST_NAME,
        "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json",
        "MF3ZU_EVIDENCE_INPUT_MANIFEST.json",
        "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json",
        "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json",
        "MF3ZU_V1R1_EVIDENCE_FREEZE_PROVENANCE.json",
        STATUS_NAME,
    ):
        path = output / name
        if path.is_file() and not path.is_symlink():
            value = strict_json(path)
            result[name] = {
                key: value.get(key)
                for key in ("status", "stage", "planned", "pass", "fail")
                if key in value
            }
    return result


def _validate_cli_output_root(output: Path) -> None:
    expected = DEFAULT_OUTPUT.resolve()
    if output != expected:
        raise RecoveryError(
            f"v1r1 output root is fixed to {rel(expected)}"
        )
    if output == PARENT_OUTPUT.resolve():
        raise RecoveryError("recovery may not write into the parent artifact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "command",
        choices=(
            "seal", "verify", "annotate-instructions", "materialize",
            "prepare-evidence", "annotate-evidence", "freeze-evidence",
            "status",
        ),
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    try:
        _validate_cli_output_root(output)
        if args.command == "seal":
            value = seal(output)
        elif args.command == "verify":
            value = verify_protocol(output)
        elif args.command == "annotate-instructions":
            value = annotate_instructions(output)
        elif args.command == "materialize":
            value = materialize_merged_view(output)
        elif args.command == "prepare-evidence":
            value = prepare_evidence(output)
        elif args.command == "annotate-evidence":
            value = annotate_evidence(output)
        elif args.command == "freeze-evidence":
            value = freeze_evidence(output)
        else:
            value = status(output)
        print(json.dumps(value, indent=2, ensure_ascii=False))
        result_status = str(value.get("status", ""))
        if result_status == "FAIL":
            return 2
        if result_status.endswith("_FAIL") or "TECHNICAL_FAIL" in result_status:
            return 3
        return 0
    except BaseException as error:
        print(
            f"MF3ZU_V1R1_RECOVERY_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
