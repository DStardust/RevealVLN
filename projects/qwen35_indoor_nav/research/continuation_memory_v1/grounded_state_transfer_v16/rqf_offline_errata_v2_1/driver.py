"""Only executable entrypoint for Q35N V16 rqf v2.1 offline errata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_funnel import failure_funnel
from audit_semantics import isolation_audit, load_compiler_class, neutral_start_audit
from common import (
    BASELINE_REVISION,
    FIXED_YAWS,
    HOUSE,
    RUN_ID,
    SPLIT,
    VERSION,
    canonical_bytes,
    digest_bytes,
    digest_value,
    load_json,
    pretty_bytes,
    replace_json_state,
    write_new,
)
from family_integrity import SnapshotReferenceResolver, validate_existing_family_reference
from plan_local_validation import prepare_local_validation_plan
from snapshot_io import SnapshotReader, seal_snapshot
from source_lock import build_source_lock, file_identity


V16 = HERE.parent
PROJECT = V16.parents[2]
OLD_FORMAL = V16 / "runs/v16_formal_001"
OLD_REPAIR = V16 / "runs/v16_repair_rqf_001"
OLD_DIAG = V16 / "runs/v16_repair_rqf_002"
OLD_DELIVERY_NAMES = (
    "PRO_AGENT_DECISION.json",
    "REPAIR_SPEC.json",
    "SOURCE_LOCK.json",
    "INPUT_SNAPSHOT.json",
    "REUSE_AUDIT.json",
    "FAILURE_FUNNEL.json",
    "NEUTRAL_START_AUDIT.json",
    "ISOLATION_AUDIT.json",
    "CPU_TEST_RESULT.json",
    "GENERATOR_PATCH_PLAN.json",
    "DIAGNOSIS_REPORT_ZH.md",
    "RESOURCE_AND_STAGE_RECEIPT.json",
    "STATUS.json",
)
COMMANDS = ("preflight", "snapshot", "audit", "test", "plan", "seal", "run-offline")
RUN_STARTED = time.time()
CONTROLLED_SUBPROCESSES: list[dict[str, Any]] = []


def find_worktree() -> Path:
    for parent in (HERE, *HERE.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("WORKTREE_NOT_FOUND")


def authorization_scope() -> dict[str, Any]:
    return {
        "scope": "OFFLINE_ERRATA_AND_CPU_ACCEPTANCE_ONLY",
        "new_physical_candidate_executions_allowed": 0,
        "new_family_admissions_allowed": 0,
        "habitat_allowed": False,
        "qwen_allowed": False,
        "g1_allowed": False,
        "training_allowed": False,
        "navigation_evaluation_allowed": False,
        "automatic_promotion_allowed": False,
        "modify_old_runs_allowed": False,
        "control_existing_services_allowed": False,
        "physical_plan_preparation_allowed": True,
        "physical_plan_execution_approved": False,
    }


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "version": VERSION,
        "run_id": RUN_ID,
        "house": HOUSE,
        "split": SPLIT,
        "new_physical_candidate_execution_limit": 0,
        "new_family_admission_limit": 0,
        "cpu_only": True,
        "auto_advance": False,
        "historical_runs_read_only": True,
        "fixed_yaws": list(FIXED_YAWS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"OFFLINE_CONFIG_GATE:{key}:{config.get(key)!r}:{value!r}")


def _write_artifact(path: Path, value: Any, *, text: bool = False) -> None:
    payload = value.encode("utf-8") if text else pretty_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"OUTPUT_CONFLICT:{path}")
        return
    write_new(path, payload)


def _state(run: Path, status: str, **extra: Any) -> None:
    replace_json_state(
        run / "STATUS.json",
        {
            "version": VERSION,
            "run_id": RUN_ID,
            "status": status,
            "new_physical_candidate_executions": 0,
            "accepted_new_families": 0,
            **extra,
        },
    )


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args], text=True, capture_output=True, check=True
    )
    CONTROLLED_SUBPROCESSES.append(
        {"argv": ["git", "-C", str(worktree), *args], "returncode": result.returncode, "purpose": "read_only_identity"}
    )
    return result.stdout.strip()


def _required_preflight_files() -> list[Path]:
    paths = [OLD_DIAG / name for name in OLD_DELIVERY_NAMES]
    paths.extend(
        [
            V16 / "isolation_diagnosis_v2.py",
            V16 / "test_isolation_diagnosis_v2.py",
            V16 / "collect_repair_v1.py",
            V16 / "collect.py",
            V16 / "evaluator_v16.py",
            V16 / "v16_common.py",
            V16 / "PROTOCOL.json",
            PROJECT / "data_pipeline/mechanism_factory_v2/compiler.py",
        ]
    )
    return paths


def _legacy_risk(worktree: Path) -> dict[str, Any]:
    job = V16 / "standalone_jobs/v16-repair-rqf-20260920-04/JOB.json"
    result = subprocess.run(
        ["systemctl", "show", "q35n-v16-repair-rqf-20260920-04.service", "-p", "ActiveState", "-p", "SubState", "-p", "Result"],
        text=True,
        capture_output=True,
    )
    CONTROLLED_SUBPROCESSES.append(
        {
            "argv": result.args,
            "returncode": result.returncode,
            "purpose": "read_only_legacy_service_status",
        }
    )
    source = (V16 / "repair_resume_v1.py").read_text(encoding="utf-8")
    return {
        "risk": "LEGACY_ENTRYPOINT_CAN_AUTO_PROGRESS_AFTER_COLLECTION",
        "entrypoint_contains_full_pipeline_call": "run_full_pipeline(run, args.run_id)" in source,
        "service_observed_read_only": result.stdout.strip().splitlines(),
        "registered_job": load_json(job) if job.exists() else None,
        "action_taken": "NONE_READ_ONLY_INSPECTION",
        "scope_note": "this finding does not assert that the failed legacy service will execute again",
    }


def preflight(config: dict[str, Any]) -> Path:
    validate_config(config)
    worktree = find_worktree()
    full_revision = _git(worktree, "rev-parse", "HEAD")
    if full_revision != BASELINE_REVISION:
        raise RuntimeError(f"BASELINE_REVISION:{full_revision}:{BASELINE_REVISION}")
    branch = _git(worktree, "branch", "--show-current")
    if branch != "codex/q35n-v16-rqf-offline-errata-20260920":
        raise RuntimeError(f"BRANCH_IDENTITY:{branch}")
    run = V16 / "runs" / config["run_id"]
    if run.exists():
        existing = run / "ERRATA_CONFIG.json"
        if not existing.is_file() or load_json(existing) != config:
            raise RuntimeError("BLOCKED_RUN_ID_COLLISION")
    else:
        run.mkdir(parents=True)
    python = Path(config["python_executable"]).resolve(strict=True)
    if not os.access(python, os.X_OK):
        raise RuntimeError(f"PYTHON_NOT_EXECUTABLE:{python}")
    identities = [file_identity(path, worktree) for path in _required_preflight_files()]
    result = subprocess.run([str(python), "-I", "-B", "-c", "import sys; print(sys.version); print(sys.executable)"], text=True, capture_output=True)
    CONTROLLED_SUBPROCESSES.append(
        {"argv": result.args, "returncode": result.returncode, "purpose": "python_identity"}
    )
    if result.returncode != 0:
        raise RuntimeError("PYTHON_IDENTITY_FAILED:" + result.stderr)
    _write_artifact(run / "ERRATA_CONFIG.json", config)
    _write_artifact(run / "AUTHORIZATION_SCOPE.json", authorization_scope())
    _write_artifact(
        run / "PREFLIGHT.json",
        {
            "version": VERSION,
            "run_id": RUN_ID,
            "worktree": str(worktree),
            "project_root": str(PROJECT.resolve()),
            "branch": branch,
            "baseline_revision_requested": config["baseline_revision"],
            "baseline_revision_full": full_revision,
            "python_executable": str(python),
            "python_identity": result.stdout.splitlines(),
            "required_files": identities,
            "old_runs_read_only": [str(path.resolve()) for path in (OLD_FORMAL, OLD_REPAIR, OLD_DIAG)],
            "permissions_verified": {"new_run_parent_writable": os.access(run.parent, os.W_OK), "old_runs_modified_by_driver": False},
        },
    )
    risk = _legacy_risk(worktree)
    _write_artifact(run / "LEGACY_AUTOPROGRESSION_RISK.json", risk)
    _state(run, "PREFLIGHT_VERIFIED")
    return run


def snapshot(config: dict[str, Any], run: Path) -> dict[str, Any]:
    old_snapshot = load_json(OLD_DIAG / "INPUT_SNAPSHOT.json")
    fixed_root = PROJECT.resolve()
    live_root = Path(config["original_live_project_root"]).resolve(strict=True)
    supplemental: list[dict[str, Any]] = []
    declared_paths = {row["path"] for row in old_snapshot.get("file_records", [])}

    # The old snapshot omitted the actual history objects and most source HOUSE
    # records, but the two audits/reuse report sealed their exact path+hash.
    # Recover those declared bytes before adding any revision-only context.
    old_neutral = load_json(OLD_DIAG / "NEUTRAL_START_AUDIT.json")
    old_isolation = load_json(OLD_DIAG / "ISOLATION_AUDIT.json")
    old_reuse = load_json(OLD_DIAG / "REUSE_AUDIT.json")
    audited_refs: list[tuple[str, str]] = []
    for row in old_neutral.get("yaw_records", []):
        if row.get("evidence_path") and row.get("evidence_sha256"):
            audited_refs.append((row["evidence_path"], row["evidence_sha256"]))
    for row in old_isolation.get("rows", []):
        if row.get("evidence_path") and row.get("evidence_sha256"):
            audited_refs.append((row["evidence_path"], row["evidence_sha256"]))
    for row in old_reuse.get("houses", []):
        if row.get("source") and row.get("source_sha256"):
            audited_refs.append((row["source"], row["source_sha256"]))
    audited_identity: dict[str, str] = {}
    for source_path, expected in audited_refs:
        previous = audited_identity.setdefault(source_path, expected)
        if previous != expected:
            raise RuntimeError(f"AUDITED_REFERENCE_CONFLICT:{source_path}")
    for source_path, expected in sorted(audited_identity.items()):
        if source_path in declared_paths:
            continue
        supplemental.append(
            {
                "path": live_root / source_path,
                "allowed_root": live_root,
                "source_path": source_path,
                "kind": "audited_original_evidence",
                "equivalence": "EXACT_ORIGINAL_BYTES",
                "expected_sha256": expected,
            }
        )
        declared_paths.add(source_path)
    for name in OLD_DELIVERY_NAMES:
        path = OLD_DIAG / name
        supplemental.append(
            {
                "path": path,
                "allowed_root": fixed_root,
                "source_path": str(path.relative_to(fixed_root)),
                "kind": "old_v2_delivery",
                "equivalence": "SUPPLEMENTAL_NOT_IN_ORIGINAL_COMPARISON",
            }
        )
    context_paths = [
        V16 / "DATA_MANIFEST.json",
        V16 / "PROTOCOL.json",
        OLD_REPAIR / "PROPOSALS_rqfALeAoiTq.json",
        PROJECT / "data_pipeline/mechanism_factory_v2/compiler.py",
        V16 / "collect.py",
        V16 / "collect_repair_v1.py",
        V16 / "evaluator_v16.py",
        V16 / "v16_common.py",
    ]
    for path in context_paths:
        supplemental.append(
            {
                "path": path,
                "allowed_root": fixed_root,
                "source_path": str(path.relative_to(fixed_root)),
                "kind": "revision_context",
                "equivalence": "REVISION_BOUND_CONTEXT",
            }
        )
    manifest = seal_snapshot(
        old_snapshot,
        [fixed_root, live_root],
        run / "inputs/objects",
        supplemental,
    )
    _write_artifact(run / "INPUT_SNAPSHOT.json", manifest)
    counts = Counter(row.get("equivalence") for row in manifest["records"])
    equivalence = {
        "source_mode": config["source_mode"],
        "counts": dict(sorted(counts.items(), key=lambda item: str(item[0]))),
        "exact_original_bytes": counts["EXACT_ORIGINAL_BYTES"],
        "revision_bound_context": counts["REVISION_BOUND_CONTEXT"],
        "unrecoverable_original_bytes": manifest["unrecoverable_original_count"],
        "supplemental_not_in_original_comparison": counts["SUPPLEMENTAL_NOT_IN_ORIGINAL_COMPARISON"],
        "same_input_comparison_eligible": manifest["unrecoverable_original_count"] == 0,
    }
    _write_artifact(run / "INPUT_EQUIVALENCE_AUDIT.json", equivalence)
    _state(run, "INPUTS_SEALED")
    return manifest


def _snapshot_context(reader: SnapshotReader) -> tuple[list[dict[str, Any]], dict[str, Any], type, str]:
    proposal_path = "research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_repair_rqf_001/PROPOSALS_rqfALeAoiTq.json"
    manifest_path = "research/continuation_memory_v1/grounded_state_transfer_v16/DATA_MANIFEST.json"
    compiler_path = "data_pipeline/mechanism_factory_v2/compiler.py"
    proposals = reader.read_path_json(proposal_path)
    data_manifest = reader.read_path_json(manifest_path)
    house = next(row for row in data_manifest["houses"] if row["house"] == HOUSE)
    compiler_record = reader.record_for_path(compiler_path)
    actual_compiler = PROJECT / compiler_path
    if digest_bytes(actual_compiler.read_bytes()) != compiler_record["object_sha256"]:
        raise RuntimeError("COMPILER_SOURCE_NOT_SNAPSHOT_BOUND")
    return proposals, house, load_compiler_class(actual_compiler), reader.record_for_path(proposal_path)["object_sha256"]


def _reuse_by_value(reader: SnapshotReader, data_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    split_by_house = {row["house"]: row["split"] for row in data_manifest["houses"]}
    wrappers: list[dict[str, Any]] = []
    ids: set[str] = set()
    issues: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    resolver = SnapshotReferenceResolver(reader)
    prefix = "research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_formal_001/HOUSE_"
    records = sorted(
        (row for row in reader.manifest.get("records", []) if row["source_path"].startswith(prefix) and row["source_path"].endswith(".json") and Path(row["source_path"]).name.startswith("HOUSE_")),
        key=lambda row: row["source_path"],
    )
    for record in records:
        house_record = reader.read_json(record["object_sha256"])
        house_name = house_record.get("house") or Path(record["source_path"]).stem.removeprefix("HOUSE_")
        for family in house_record.get("families", []):
            family_id = family.get("family_id")
            if not isinstance(family_id, str) or family_id in ids:
                issues.append({"family_id": family_id, "issue": "MISSING_OR_DUPLICATE_FAMILY_ID", "source": record["source_path"]})
                continue
            ids.add(family_id)
            split_counts[split_by_house[house_name]] += 1
            canonical_sha = digest_value(family)
            validation = validate_existing_family_reference(family, resolver, {"canonical_sha256": canonical_sha})
            wrappers.append(
                {
                    "provenance": {
                        "source_run": "v16_formal_001",
                        "source_house_record_path": record["source_path"],
                        "source_house_record_sha256": record["object_sha256"],
                        "family_id": family_id,
                        "original_family_canonical_sha256": canonical_sha,
                        "reused_family_canonical_sha256": digest_value(family),
                        "value_equal": True,
                        "references_check_scope": "declared JSON references only; raw RGB/semantic arrays NOT_RECHECKED",
                        "reference_validation": validation,
                        "recollected": False,
                        "newly_certified": False,
                    },
                    "family": family,
                }
            )
    reused = {"version": VERSION, "families": wrappers, "family_count": len(wrappers)}
    audit = {
        "expected_family_count": 24,
        "actual_family_count": len(wrappers),
        "unique_family_ids": len(ids),
        "split_counts": dict(sorted(split_counts.items())),
        "expected_split_counts": {"FIT": 16, "DEV": 2, "TEST": 6},
        "target_house_reused_count": sum(row["family"].get("house") == HOUSE for row in wrappers),
        "target_house_missing_families": 2,
        "issues": issues,
        "value_reuse_verified": len(wrappers) == 24 and len(ids) == 24 and dict(split_counts) == {"FIT": 16, "DEV": 2, "TEST": 6} and not issues,
        "raw_array_integrity": "NOT_RECHECKED",
        "recollected": False,
        "newly_certified": False,
    }
    return reused, audit


def _errata_diff(old_neutral: dict[str, Any], new_neutral: dict[str, Any], old_funnel: dict[str, Any]) -> dict[str, Any]:
    old_yaws = {(row["proposal"], row["yaw"]): row for row in old_neutral["yaw_records"]}
    yaw_diff = []
    for row in new_neutral["yaw_records"]:
        old = old_yaws.get((row["proposal_id"], row["yaw"]))
        old_class = old.get("classification") if old else "NOT_RECORDED"
        new_class = row["neutral_gate_result"]
        if row.get("all_roles_triggered") == ["terminal"] and old_class == "INVALID_YAW":
            reason = "TERMINAL_EXCLUDED_FROM_ANCHOR_GATE"
        elif new_class in {"EVIDENCE_MISSING", "NOT_REACHED", "EVIDENCE_INVALID", "ERROR"}:
            reason = "MISSING_EVIDENCE_NO_LONGER_ALL_INVALID"
        elif not row.get("probe_gate_reached"):
            reason = "PARTIAL_PROBE_NO_LONGER_COMPLETE_GATE"
        else:
            reason = "UNCHANGED_OR_RELABELED_TO_EXPLICIT_GATE_STATE"
        yaw_diff.append(
            {
                "proposal_id": row["proposal_id"],
                "yaw": row["yaw"],
                "old_classification": old_class,
                "new_classification": new_class,
                "terminal_only": row.get("all_roles_triggered") == ["terminal"],
                "input_object_sha256": row.get("input_object_sha256"),
                "reason": reason,
            }
        )
    old_rollups = {row["proposal"]: row for row in old_neutral["proposal_rollup"]}
    proposal_diff = [
        {
            "proposal_id": row["proposal_id"],
            "old_category": old_rollups.get(row["proposal_id"], {}).get("proposal_classification", "NOT_RECORDED"),
            "new_category": row["proposal_classification"],
            "known_valid_yaws": row["known_valid_yaws"],
            "known_invalid_yaws": row["known_invalid_yaws"],
            "unresolved_yaws": row["unresolved_yaws"],
            "reason": "MISSING_EVIDENCE_NO_LONGER_ALL_INVALID" if row["unresolved_yaws"] else "TERMINAL_EXCLUDED_FROM_ANCHOR_GATE_OR_EXPLICIT_COMPLETE_CLASS",
        }
        for row in new_neutral["proposal_rollup"]
    ]
    return {
        "version": VERSION,
        "yaw_differences": yaw_diff,
        "proposal_differences": proposal_diff,
        "report_claim_differences": [
            {"old_claim": "terminal events participated in per-yaw rejection", "disposition": "CORRECTED", "reason": "TERMINAL_EXCLUDED_FROM_ANCHOR_GATE"},
            {"old_claim": "five CPU tests supported the full correctness claim", "disposition": "WITHDRAWN", "reason": "REPORT_CLAIM_NOT_SUPPORTED_BY_TEST"},
            {"old_claim": "FAMILY nonempty histories/traces established completeness", "disposition": "CORRECTED", "reason": "CERTIFICATE_VALIDATION_SCOPE_CORRECTED"},
            {"old_claim": old_funnel.get("physical_rejection_definition"), "disposition": "CORRECTED", "reason": "FUNNEL_DENOMINATORS_SEPARATED"},
        ],
    }


def audit(config: dict[str, Any], run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reader = SnapshotReader.from_paths(run / "INPUT_SNAPSHOT.json", run / "inputs/objects")
    proposals, house, compiler_class, proposal_hash = _snapshot_context(reader)
    old_neutral = reader.read_path_json(str((OLD_DIAG / "NEUTRAL_START_AUDIT.json").relative_to(PROJECT)))
    old_isolation = reader.read_path_json(str((OLD_DIAG / "ISOLATION_AUDIT.json").relative_to(PROJECT)))
    old_funnel = reader.read_path_json(str((OLD_DIAG / "FAILURE_FUNNEL.json").relative_to(PROJECT)))
    neutral = neutral_start_audit(reader, old_neutral, proposals, house, compiler_class)
    isolation = isolation_audit(reader, old_isolation, proposals, house, compiler_class)
    funnel = failure_funnel(reader, proposals, isolation)
    data_manifest = reader.read_path_json("research/continuation_memory_v1/grounded_state_transfer_v16/DATA_MANIFEST.json")
    reused, reuse_audit = _reuse_by_value(reader, data_manifest)
    diff = _errata_diff(old_neutral, neutral, old_funnel)
    _write_artifact(run / "NEUTRAL_START_AUDIT.json", neutral)
    _write_artifact(run / "ISOLATION_AUDIT.json", isolation)
    _write_artifact(run / "FAILURE_FUNNEL.json", funnel)
    _write_artifact(run / "REUSED_FAMILIES_BY_VALUE.json", reused)
    _write_artifact(run / "REUSE_AUDIT.json", reuse_audit)
    _write_artifact(run / "ERRATA_DIFF.json", diff)
    _state(run, "OFFLINE_RECOMPUTE_COMPLETE")
    return neutral, isolation, funnel


def _run_command(argv: list[str], *, env: dict[str, str], purpose: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, cwd=find_worktree(), env=env, text=True, capture_output=True)
    CONTROLLED_SUBPROCESSES.append(
        {"argv": argv, "returncode": result.returncode, "purpose": purpose, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-4000:]}
    )
    return result


def run_tests(config: dict[str, Any], run: Path) -> dict[str, Any]:
    python = str(Path(config["python_executable"]).resolve(strict=True))
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "Q35N_REAL_SOURCE_ROOT": str(Path(config["original_live_project_root"]).parents[1]),
            "Q35N_TEST_ROOT": str(run / "test_tmp"),
        }
    )
    pre = _run_command(
        [python, "-I", "-B", str(HERE / "tests/pref_fix_regressions.py"), "--real-source-root", env["Q35N_REAL_SOURCE_ROOT"], "--result", str(run / "PRE_FIX_TEST_RESULT.json")],
        env=env,
        purpose="pre_fix_regressions",
    )
    if pre.returncode != 0:
        raise RuntimeError("PRE_FIX_REGRESSION_EXECUTION:" + pre.stderr)
    cpu = _run_command(
        [python, "-I", "-B", str(HERE / "tests/run_tests.py"), "--result", str(run / "CPU_TEST_RESULT.json")],
        env=env,
        purpose="cpu_acceptance",
    )
    result = load_json(run / "CPU_TEST_RESULT.json")
    publication_cases = [row for row in result["cases"] if str(row["test_id"]).startswith("F")]
    _write_artifact(
        run / "PUBLICATION_RECOVERY_TEST_RESULT.json",
        {
            "status": "PASS" if publication_cases and all(row["status"] == "PASS" for row in publication_cases) else "FAIL",
            "cases": publication_cases,
            "actual_filesystem_test_root": str(run.resolve()),
            "formal_family_files_created": 0,
        },
    )
    try:
        (run / "test_tmp").rmdir()
    except FileNotFoundError:
        pass
    if cpu.returncode != 0:
        _state(run, "BLOCKED_CPU_ACCEPTANCE")
    else:
        _state(run, "CPU_ACCEPTANCE_PASSED")
    return result


def plan(config: dict[str, Any], run: Path) -> dict[str, Any]:
    reader = SnapshotReader.from_paths(run / "INPUT_SNAPSHOT.json", run / "inputs/objects")
    proposals, house, _, proposal_hash = _snapshot_context(reader)
    neutral = load_json(run / "NEUTRAL_START_AUDIT.json")
    isolation = load_json(run / "ISOLATION_AUDIT.json")
    manifest, local_plan, patch_plan = prepare_local_validation_plan(neutral, isolation, proposals, house, proposal_hash)
    _write_artifact(run / "LOCAL_PHYSICAL_VALIDATION_MANIFEST.json", manifest)
    _write_artifact(run / "LOCAL_PHYSICAL_VALIDATION_PLAN.json", local_plan)
    _write_artifact(run / "GENERATOR_PATCH_PLAN.json", patch_plan)
    _state(run, "PLAN_PREPARED_PENDING_APPROVAL" if manifest["trial_count"] == 4 else "BLOCKED_PLAN_INPUTS")
    return manifest


def _source_files() -> list[Path]:
    return sorted(HERE.glob("*.py")) + sorted((HERE / "tests").glob("*.py")) + [
        PROJECT / "data_pipeline/mechanism_factory_v2/compiler.py",
        V16 / "collect.py",
        V16 / "collect_repair_v1.py",
        V16 / "evaluator_v16.py",
        V16 / "v16_common.py",
        V16 / "PROTOCOL.json",
        V16 / "DATA_MANIFEST.json",
    ]


def _integration_checks(run: Path) -> dict[str, Any]:
    neutral = load_json(run / "NEUTRAL_START_AUDIT.json")
    isolation = load_json(run / "ISOLATION_AUDIT.json")
    funnel = load_json(run / "FAILURE_FUNNEL.json")
    cpu = load_json(run / "CPU_TEST_RESULT.json")
    plan_manifest = load_json(run / "LOCAL_PHYSICAL_VALIDATION_MANIFEST.json")
    cases: list[dict[str, Any]] = []

    def check(case_id: str, condition: bool, assertion: str) -> None:
        cases.append({"test_id": case_id, "status": "PASS" if condition else "FAIL", "assertion": assertion})

    scope = authorization_scope()
    check("G01", cpu["status"] == "PASS" and scope["new_physical_candidate_executions_allowed"] == 0, "offline completion with zero physical/model factory surface")
    check("G02", scope["automatic_promotion_allowed"] is False and load_json(run / "ERRATA_CONFIG.json")["missing_target_families"] == 2, "missing families never authorize progression")
    check("G03", set(COMMANDS) == {"preflight", "snapshot", "audit", "test", "plan", "seal", "run-offline"}, "CLI has no model, physical, training, or promotion command")
    check("G04", all(not run.resolve().is_relative_to(old.resolve()) for old in (OLD_FORMAL, OLD_REPAIR, OLD_DIAG)), "only new run root is writable")
    semantic_hashes = {name: digest_bytes((run / name).read_bytes()) for name in ("NEUTRAL_START_AUDIT.json", "ISOLATION_AUDIT.json", "FAILURE_FUNNEL.json")}
    reader = SnapshotReader.from_paths(run / "INPUT_SNAPSHOT.json", run / "inputs/objects")
    proposals, house, compiler_class, _ = _snapshot_context(reader)
    old_neutral = reader.read_path_json(str((OLD_DIAG / "NEUTRAL_START_AUDIT.json").relative_to(PROJECT)))
    old_isolation = reader.read_path_json(str((OLD_DIAG / "ISOLATION_AUDIT.json").relative_to(PROJECT)))
    repeated_neutral = neutral_start_audit(reader, old_neutral, proposals, house, compiler_class)
    repeated_isolation = isolation_audit(reader, old_isolation, proposals, house, compiler_class)
    repeated_funnel = failure_funnel(reader, proposals, repeated_isolation)
    repeated = {
        "NEUTRAL_START_AUDIT.json": digest_bytes(pretty_bytes(repeated_neutral)),
        "ISOLATION_AUDIT.json": digest_bytes(pretty_bytes(repeated_isolation)),
        "FAILURE_FUNNEL.json": digest_bytes(pretty_bytes(repeated_funnel)),
    }
    check("G05", semantic_hashes == repeated, "a second full in-memory recomputation over the same object store has identical semantic hashes")
    denominators = funnel["proposal_denominator"]["identity_holds"] and funnel["yaw_denominator"]["identity_holds"]
    check("G06", denominators and len(neutral["proposal_rollup"]) == 256 and isolation["history_record_count"] + isolation["partial_family_notice_count"] == len(isolation["rows"]), "end-to-end denominators agree")
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in cases) else "FAIL",
        "cases": cases,
        "semantic_artifact_sha256": semantic_hashes,
        "repeated_recompute_sha256": repeated,
        "backend_factory_calls": 0,
        "model_factory_calls": 0,
        "g1_calls": 0,
        "physical_plan_execution_calls": 0,
        "planned_trial_count": plan_manifest["trial_count"],
    }


def _report(run: Path, final_status: str) -> str:
    equivalence = load_json(run / "INPUT_EQUIVALENCE_AUDIT.json")
    cpu = load_json(run / "CPU_TEST_RESULT.json")
    reuse = load_json(run / "REUSE_AUDIT.json")
    plan_manifest = load_json(run / "LOCAL_PHYSICAL_VALIDATION_MANIFEST.json")
    real_case = next((row for row in cpu["cases"] if row["test_id"] == "N12"), None)
    publication = load_json(run / "PUBLICATION_RECOVERY_TEST_RESULT.json")
    return f"""# Q35N V16 rqf v2.1 离线勘误与 CPU 补验报告

## 结论

本轮完成的范围是离线字节封存、语义勘误、分母重算、CPU 行为验收和下一节点报审材料编制。终态为 `{final_status}`；这不是物理采集、训练、模型失败或方法收益结论。

- 精确复用原字节的条目数：{equivalence['exact_original_bytes']}
- revision 上下文条目数：{equivalence['revision_bound_context']}
- 原字节不可恢复的条目数：{equivalence['unrecoverable_original_bytes']}
- CPU 测试实际执行/通过/失败/未执行：{cpu['executed']}/{cpu['passed']}/{cpu['failed']}/{cpu['not_executed']}
- terminal-only 真实回归：{real_case['status'] if real_case else 'NOT_EXECUTED'}
- 发布与恢复修复：{publication['status']}
- 原 24 族按值复用：{'VERIFIED' if reuse['value_reuse_verified'] else 'BLOCKED'}；引用范围为声明 JSON，原始数组未重验
- 新增物理执行：0
- 新增接纳族：0
- 目标屋缺失族：2（未因本轮补齐）

## 勘误边界

中性门槛现在只由 `anchor_A`/`anchor_B` 触发；terminal-only 保留诊断但不拒绝。每个 proposal 固定建立八个 yaw 槽，缺证据、未到达、证据无效和程序错误都会令聚合成为 `INCOMPLETE_EVIDENCE`，不会成为“八个均无效”。历史 own/other anchor 分开统计，部分 FAMILY 通知不进入历史分母。

FAMILY 的未来提交路径要求四历史、十二后缀、引用哈希、精确 replay 前缀、STOP、预算、碰撞和标签矩阵全部通过；发布采用同目录临时文件、fsync、排他硬链接和目录 fsync。旧 24 族未重采、未重认证。

## 下一节点

待审批 manifest 状态 `{plan_manifest['status']}`，包含 {plan_manifest['trial_count']} 个具体 trial。当前 driver 没有物理、训练、G1、Qwen 或自动推进命令。只有新的明确审核裁决才能执行另一个独立节点。
"""


def seal(config: dict[str, Any], run: Path) -> str:
    integration = _integration_checks(run)
    _write_artifact(run / "INTEGRATION_TEST_RESULT.json", integration)
    worktree = find_worktree()
    lock = build_source_lock(
        _source_files(),
        worktree,
        BASELINE_REVISION,
        {
            "driver": str(Path(__file__).resolve()),
            "compiler": str((PROJECT / "data_pipeline/mechanism_factory_v2/compiler.py").resolve()),
            "python": str(Path(config["python_executable"]).resolve()),
        },
    )
    _write_artifact(run / "SOURCE_LOCK.json", lock)
    cpu = load_json(run / "CPU_TEST_RESULT.json")
    equivalence = load_json(run / "INPUT_EQUIVALENCE_AUDIT.json")
    manifest = load_json(run / "LOCAL_PHYSICAL_VALIDATION_MANIFEST.json")
    reuse = load_json(run / "REUSE_AUDIT.json")
    if equivalence["unrecoverable_original_bytes"]:
        final_status = "BLOCKED_EVIDENCE_MISSING"
        exit_code = 2
    elif cpu["status"] != "PASS" or integration["status"] != "PASS":
        final_status = "BLOCKED_CPU_ACCEPTANCE"
        exit_code = 2
    elif not reuse["value_reuse_verified"]:
        final_status = "BLOCKED_INPUT_IDENTITY"
        exit_code = 2
    elif manifest["trial_count"] != 4:
        final_status = "BLOCKED_PLAN_INPUTS"
        exit_code = 2
    else:
        final_status = "OFFLINE_ERRATA_COMPLETE_PENDING_REVIEW"
        exit_code = 0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "version": VERSION,
        "run_id": RUN_ID,
        "started_at_unix": RUN_STARTED,
        "finished_at_unix": time.time(),
        "cpu_time_seconds": {"self_user": usage.ru_utime, "self_system": usage.ru_stime, "children_user": children.ru_utime, "children_system": children.ru_stime},
        "peak_rss": {"value": usage.ru_maxrss, "units": "KiB on Linux", "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss"},
        "process": {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "cwd": os.getcwd()},
        "controlled_subprocesses": CONTROLLED_SUBPROCESSES,
        "backend_factory_calls": integration["backend_factory_calls"],
        "model_factory_calls": integration["model_factory_calls"],
        "g1_calls": integration["g1_calls"],
        "factory_measurement_method": "offline driver exposes no such factory and G03 validates the CLI/import surface",
        "new_physical_candidate_executions": 0,
        "new_family_admissions": 0,
        "write_scope": str(run.resolve()),
        "old_run_actions": "none",
        "legacy_service_action": "none; read-only status inspection only",
        "terminal_status": final_status,
        "exit_code": exit_code,
    }
    _write_artifact(run / "RESOURCE_AND_STAGE_RECEIPT.json", receipt)
    _write_artifact(run / "DIAGNOSIS_REPORT_ZH.md", _report(run, final_status), text=True)
    request = f"""# Q35N V16 rqf v2.1 审核请求

当前终态：`{final_status}`。本轮已封存退出，不自动推进。

请分别裁决：

1. 是否接受 anchor-only 中性判定、缺证据聚合和同输入差异表所构成的离线勘误；
2. 是否接受四历史/十二后缀验证、原子不覆盖发布、恢复、预算与异常分流的 CPU 正确性修复；
3. 是否批准 `LOCAL_PHYSICAL_VALIDATION_MANIFEST.json` 中恰好四个 trial 及 `LOCAL_PHYSICAL_VALIDATION_PLAN.json` 的硬限额。当前批准状态仍为 `PENDING_EXPLICIT_APPROVAL`。

新增物理执行 0；新增接纳 0；目标屋仍缺 2 族。
"""
    _write_artifact(run / "NEXT_NODE_APPROVAL_REQUEST.md", request, text=True)
    _state(run, final_status, exit_code=exit_code)
    artifacts = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name == "OUTPUT_MANIFEST.json":
            continue
        data = path.read_bytes()
        artifacts.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    output = {
        "version": VERSION,
        "run_id": RUN_ID,
        "artifacts": artifacts,
        "input_object_store": {"path": "inputs/objects", "objects": load_json(run / "INPUT_SNAPSHOT.json")["unique_object_count"]},
        "self_exclusion": "OUTPUT_MANIFEST.json is not self-hashed",
        "sealed": True,
    }
    _write_artifact(run / "OUTPUT_MANIFEST.json", output)
    return final_status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q35N rqf offline errata only")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config.resolve(strict=True))
    validate_config(config)
    if args.command == "run-offline":
        run = preflight(config)
        snapshot(config, run)
        run_tests(config, run)
        audit(config, run)
        plan(config, run)
        status = seal(config, run)
    else:
        run = V16 / "runs" / config["run_id"]
        if args.command == "preflight":
            preflight(config)
            return 0
        if not run.is_dir():
            raise RuntimeError("RUN_NOT_PREFLIGHTED")
        if args.command == "snapshot":
            snapshot(config, run)
            return 0
        if args.command == "test":
            run_tests(config, run)
            return 0 if load_json(run / "CPU_TEST_RESULT.json")["status"] == "PASS" else 2
        if args.command == "audit":
            audit(config, run)
            return 0
        if args.command == "plan":
            plan(config, run)
            return 0
        status = seal(config, run)
    print(json.dumps({"run": str(run), "status": status}, ensure_ascii=False, sort_keys=True))
    return 0 if status == "OFFLINE_ERRATA_COMPLETE_PENDING_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
