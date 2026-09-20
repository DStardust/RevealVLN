"""Offline, CPU-only diagnosis for the rqf repair shortfall.

This module deliberately has no Habitat, torch, CUDA, Qwen, training, or
pipeline entry point.  It consumes a sealed snapshot of already written v1
evidence and emits a diagnosis-only run.  It is an independent implementation
so that the failed v1 collector and its source lock remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
RUNS = HERE / "runs"
VERSION = "v16_rqf_isolation_diagnosis_v2"
RUN_ID = "v16_repair_rqf_002"
HOUSE = "rqfALeAoiTq"
BASELINE_COMMIT = "e4a8b2745cd077c690be33081733b5a82ee04f3c"
SEE2_PIXELS = 256
HISTORY_CAP = 240
FULL_DECISION_CAP = 500
YAW_SET = list(range(0, 24, 3))


class Rejected(RuntimeError):
    """Explicit physical rejection; never use this for a program error."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(LINE.resolve()))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_json_or_none(path: Path) -> Any | None:
    try:
        return load_json(path)
    except (OSError, ValueError, TypeError):
        return None


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, value: Any) -> None:
    """Create immutable JSON evidence; an existing unequal file is fatal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"IMMUTABLE_OUTPUT_MISMATCH:{path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_text_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"IMMUTABLE_OUTPUT_MISMATCH:{path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def validate_family_for_publish(family: Any) -> None:
    if not isinstance(family, dict):
        raise ValueError("FAMILY_NOT_OBJECT")
    required = {"family_id", "house", "split", "roles", "compiler", "histories",
                "traces", "content_root", "proposal", "training_admission"}
    missing = sorted(required - set(family))
    if missing:
        raise ValueError("FAMILY_INCOMPLETE:" + ",".join(missing))
    if not isinstance(family["content_root"], str) or not family["content_root"]:
        raise ValueError("FAMILY_CONTENT_ROOT")


def publish_family_once(path: Path, family: Any) -> None:
    """The v2 success path: validate, then one exclusive publication only."""
    validate_family_for_publish(family)
    if path.exists():
        raise FileExistsError(f"FAMILY_ALREADY_PUBLISHED:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(family, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, Rejected):
        return "PHYSICAL_REJECTED"
    if isinstance(exc, (FileNotFoundError, json.JSONDecodeError)):
        return "EVIDENCE_MISSING"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "INTERRUPTED"
    return "ERROR"


def diagnosis_only_guard() -> dict[str, Any]:
    """Machine-readable proof of the intentionally narrow execution surface."""
    source = Path(__file__).read_text(encoding="utf-8")
    # Build the tokens in pieces so this self-check does not match its own
    # literal tuple.  Generic words such as ``torch`` are intentionally not
    # searched: the guard is about executable calls, not prose.
    forbidden = ("repair_resume_v1" + ".main(", "promote_pilot" + ".py",
                 "pipeline.py " + "--phase all", "run_full_pipeline" + "(",
                 "Habitat" + "Backend(", "import " + "torch", "torch" + ".", "cu" + "da")
    hits = {token: source.count(token) for token in forbidden if token in source}
    if hits:
        raise RuntimeError("DIAGNOSIS_ENTRYPOINT_FORBIDDEN_TOKEN:" + repr(hits))
    return {"forbidden_tokens_checked": list(forbidden), "hits": {},
            "creates_formal_family_json": False, "loads_model": False,
            "new_physical_candidate_executions": 0}


def import_compiler() -> type:
    path = LINE / "data_pipeline" / "mechanism_factory_v2" / "compiler.py"
    spec = importlib.util.spec_from_file_location("q35n_v2_compiler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"COMPILER_IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Compiler


def proposal_tasks(roles: dict[str, dict[str, Any]], split: str) -> dict[str, dict[str, str]]:
    terminal = roles["terminal"]
    terminal_text = f"the {terminal['raw_match']['value']} in the {terminal['room']}"
    result: dict[str, dict[str, str]] = {}
    for task_id, anchor in (("task_A", "anchor_A"), ("task_B", "anchor_B")):
        spec = roles[anchor]
        anchor_text = f"the {spec['raw_match']['value']} in the {spec['room']}"
        if split == "FIT":
            instruction = f"First see {anchor_text} in two consecutive observations, then see {terminal_text} in two consecutive observations, and stop immediately."
        elif split == "DEV":
            instruction = f"Before stopping at {terminal_text}, establish a two-observation sighting of {anchor_text}. Stop only on a subsequent two-observation sighting of {terminal_text}."
        else:
            instruction = f"Complete these sightings in order: {anchor_text}, followed by {terminal_text}. Each sighting requires two consecutive views. End by stopping on the final sighting."
        result[task_id] = {"anchor": anchor, "terminal": "terminal", "instruction": instruction}
    return result


def make_compiler(proposal: dict[str, Any], house: dict[str, Any]) -> Any:
    indices = proposal.get("role_indices", [])
    if len(indices) != 3:
        raise ValueError("PROPOSAL_ROLE_INDICES")
    roles: dict[str, dict[str, Any]] = {}
    eligible: dict[str, list[int]] = {}
    for role, index in zip(("anchor_A", "anchor_B", "terminal"), indices):
        row = house["roles"][int(index)]
        roles[role] = row["spec"]
        eligible[role] = list(row.get("eligible", []))
    config = {"roles": {k: [v["mpcat40"], v["room"]] for k, v in roles.items()},
              "tasks": proposal_tasks(roles, house["split"]), "eligible": eligible}
    Compiler = import_compiler()
    return Compiler(**config)


def pixels(observation: dict[str, Any], instance: int) -> int:
    values = observation.get("pixels", {}) if isinstance(observation, dict) else {}
    return int(values.get(str(instance), values.get(instance, 0)) or 0)


def hit_details(observations: list[dict[str, Any]], compiler: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return compiler events and exact same-instance adjacent pixel evidence."""
    events = compiler.atoms(observations)
    hits: list[dict[str, Any]] = []
    for step, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        for role, ids in event.items():
            if not ids:
                continue
            rows = []
            for instance in ids:
                current = pixels(observations[step], int(instance))
                previous = pixels(observations[step - 1], int(instance)) if step else 0
                rows.append({"instance": int(instance), "previous_pixels": previous,
                             "current_pixels": current, "same_instance": True,
                             "see2": bool(previous >= SEE2_PIXELS and current >= SEE2_PIXELS)})
            hits.append({"step": step, "role": role, "instances": rows})
    return events, hits


def trace_summary(path: Path, proposal: dict[str, Any], house: dict[str, Any],
                  history_id: str, original_error: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "proposal": proposal.get("id"), "yaw": path.parent.name,
        "history_id": history_id, "evidence_path": rel(path),
        "evidence_sha256": sha256_file(path), "original_error": original_error,
        "stage_boundary_status": "UNRESOLVED",
        "first_contamination_stage": "UNRESOLVED",
        "stage_reason": "saved v1 trace has no machine-readable phase boundary; route shape is not used to infer a stage",
    }
    trace = load_json_or_none(path)
    if not isinstance(trace, dict) or not isinstance(trace.get("observations"), list):
        record.update({"offline_recompute": "EVIDENCE_MISSING", "evidence_status": "EVIDENCE_MISSING"})
        return record
    try:
        compiler = make_compiler(proposal, house)
        observations = trace["observations"]
        events, hits = hit_details(observations, compiler)
        own = "anchor_A" if history_id.startswith("H_A") else "anchor_B"
        other = "anchor_B" if own == "anchor_A" else "anchor_A"
        own_hits = [row for row in hits if row["role"] == own]
        other_hits = [row for row in hits if row["role"] == other]
        record.update({
            "history_observation_count": len(observations),
            "history_action_count": len(trace.get("actions", [])),
            "collisions": trace.get("collisions"),
            "trace_complete": trace.get("complete"),
            "target_anchor": own,
            "other_anchor": other,
            "target_anchor_first_step": own_hits[0]["step"] if own_hits else None,
            "other_anchor_first_step": other_hits[0]["step"] if other_hits else None,
            "target_anchor_first_trigger": own_hits[0] if own_hits else None,
            "other_anchor_first_trigger": other_hits[0] if other_hits else None,
            "compiler_event_count": sum(1 for e in events if any(e.get(r) for r in (own, other))),
            "see2_rule": "Compiler.atoms; same eligible instance >=256 pixels in both adjacent actual observations",
        })
        if not own_hits:
            record["offline_recompute"] = "OWN_ANCHOR_NOT_SEEN"
        elif other_hits:
            record["offline_recompute"] = "RECOMPUTED_OTHER_ANCHOR_PRESENT"
        else:
            record["offline_recompute"] = "ISOLATED_FOR_SAVED_TRACE"
        record["evidence_status"] = "READ"
    except Exception as exc:  # diagnostic failure is not a physical rejection
        record.update({"offline_recompute": "ERROR", "evidence_status": "ERROR",
                       "diagnostic_error": repr(exc), "traceback": traceback.format_exc(limit=3)})
    return record


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"_line": line_no, "_parse_error": True})
                continue
            if isinstance(value, dict):
                value["_line"] = line_no
                rows.append(value)
    return rows


def yaw_search_rows(folder: Path) -> list[dict[str, Any]]:
    value = load_json_or_none(folder / "REPAIR_YAW_SEARCH.json")
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in ("attempts", "yaws", "rows"):
            if isinstance(value.get(key), list):
                return [x for x in value[key] if isinstance(x, dict)]
    rows = parse_jsonl(folder / "REPAIR_YAW_ATTEMPTS.jsonl")
    return [x for x in rows if x.get("status") in {"REJECTED", "CERTIFIED"} or "error" in x]


def row_yaw(row: dict[str, Any], folder: Path) -> str | None:
    value = row.get("yaw", row.get("initial_yaw", row.get("selected_yaw")))
    if value is None:
        match = re.search(r"yaw_(\d+)", str(folder))
        return match.group(1) if match else None
    try:
        return f"{int(value):02d}"
    except (TypeError, ValueError):
        return str(value)


def original_error_for_yaw(rows: list[dict[str, Any]], yaw: str) -> str | None:
    for row in rows:
        if row_yaw(row, Path("yaw_" + yaw)) == yaw:
            for key in ("error", "reason", "result", "status"):
                if key in row:
                    value = row[key]
                    return value if isinstance(value, str) else repr(value)
    return None


def selected_input_files(old_run: Path, repair_run: Path) -> list[Path]:
    paths: list[Path] = []
    old_names = ["STATUS.json", "DATA_SHORTFALL.json", "HOUSE_rqfALeAoiTq.json",
                 "PROTOCOL.json", "COLLECTION_PROGRESS.json", "COLLECTION_ATTEMPTS.jsonl",
                 "FAILURES.jsonl", "SOURCE_LOCK.json"]
    repair_names = ["STATUS.json", "COLLECTION_PROGRESS.json", "COLLECTION_ATTEMPTS.jsonl",
                    "PROTOCOL.json", "REPAIR_SPEC.json", "SOURCE_LOCK.json", "GPU_ASSIGNMENT.json"]
    for name in old_names:
        paths.append(old_run / name)
    for name in repair_names:
        paths.append(repair_run / name)
    # All proposal-level failure/search/initial-probe files are small evidence
    # records.  Raw continuation contents and the content-addressed array store
    # are intentionally excluded from the input seal and never copied.
    for root in (old_run / "proposals", repair_run / "proposals"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.name in {"FAILURE.json", "REPAIR_YAW_SEARCH.json",
                                                   "REPAIR_YAW_ATTEMPTS.jsonl", "INITIAL_VIEW_PROBE.json",
                                                   "FAMILY.json"}):
                paths.append(path)
    result = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file() and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def file_record(path: Path, cutoff: float, log_limit: int = 4096) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {"path": rel(path), "bytes": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path),
                              "included_at_or_before_cutoff": stat.st_mtime <= cutoff}
    if path.suffix in {".log", ".jsonl"} or path.name.endswith(".log"):
        size = stat.st_size
        with path.open("rb") as stream:
            first = stream.read(log_limit)
            if size > log_limit:
                stream.seek(max(0, size - log_limit))
                last = stream.read(log_limit)
            else:
                last = first
        record["log_truncation"] = {"mode": "first_and_last_bytes", "limit": log_limit,
                                    "first_bytes": len(first), "last_bytes": len(last),
                                    "start_offset": max(0, size - log_limit),
                                    "end_offset": size}
    return record


def input_snapshot(old_run: Path, repair_run: Path, cutoff: float) -> dict[str, Any]:
    records = []
    late = []
    for path in selected_input_files(old_run, repair_run):
        try:
            record = file_record(path, cutoff)
        except OSError as exc:
            late.append({"path": rel(path), "error": repr(exc)})
            continue
        records.append(record)
        if not record["included_at_or_before_cutoff"]:
            late.append({"path": record["path"], "mtime_ns": record["mtime_ns"],
                         "reason": "written after explicit snapshot cutoff; excluded from completed evidence"})
    old_progress = load_json_or_none(old_run / "COLLECTION_PROGRESS.json") or {}
    repair_progress = load_json_or_none(repair_run / "COLLECTION_PROGRESS.json") or {}
    return {
        "version": VERSION, "run_id": RUN_ID, "snapshot_cutoff_unix": cutoff,
        "snapshot_cutoff_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff)),
        "source_runs": {"certified_source": rel(old_run), "failure_source": rel(repair_run)},
        "completed_proposal_inputs": {
            "v16_formal_001_house_rqf_attempts": int((load_json_or_none(old_run / "HOUSE_rqfALeAoiTq.json") or {}).get("attempts", 0)),
            "v16_repair_rqf_001_progress": repair_progress,
            "v1_completed_rejection_rows_at_cutoff": sum(1 for row in parse_jsonl(repair_run / "COLLECTION_ATTEMPTS.jsonl")
                                                          if row.get("status") == "REJECTED" and row.get("unix", 0) <= cutoff),
            "v1_partial_or_in_progress_not_completed": True,
        },
        "source_progress_at_cutoff": {"old": old_progress, "repair": repair_progress},
        "file_records": records,
        "late_or_unsealed_files": late,
        "excluded_scopes": [
            {"scope": rel(old_run / "content"), "reason": "content-addressed arrays are not needed for the compiler audit and are not copied"},
            {"scope": rel(repair_run / "content"), "reason": "no model/data generation in diagnosis"},
            {"scope": "raw continuation trace files not needed for the two requested audits", "reason": "not copied; H_*_DISCOVERY and INITIAL_VIEW_PROBE evidence used by the audits carries its own path and sha256 in ISOLATION_AUDIT/NEUTRAL_START_AUDIT; not silently treated as missing"},
        ],
        "audited_evidence_hash_source": {"neutral_start": "NEUTRAL_START_AUDIT.json", "history_isolation": "ISOLATION_AUDIT.json"},
        "read_policy": "completed files are sealed by mtime and sha256; active directories were not chmod/moved/renamed/locked",
    }


def completed_family(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        validate_family_for_publish(value)
    except Exception:
        return False
    return bool(value.get("traces")) and bool(value.get("histories"))


def reuse_audit(old_run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total = 0
    for path in sorted(old_run.glob("HOUSE_*.json")):
        house = path.stem[len("HOUSE_"):]
        value = load_json_or_none(path)
        if not isinstance(value, dict):
            continue
        families = value.get("families", [])
        valid = [f for f in families if completed_family(f)]
        total += len(valid)
        rows.append({"house": house, "source": rel(path), "source_sha256": sha256_file(path),
                     "listed_families": len(families), "validated_complete_families": len(valid),
                     "complete_flag": value.get("complete"),
                     "family_ids": [f.get("family_id") for f in valid],
                     "reuse_mode": "by_value_reference_only", "recollected": False})
    expected_houses = {h["house"]: h for h in manifest.get("houses", []) if h.get("split") in {"FIT", "DEV", "TEST"}}
    return {"source_run": rel(old_run), "source_version": "v16_formal_001",
            "expected_complete_family_count": 24, "validated_complete_family_count": total,
            "target_house": HOUSE, "target_house_reused_count": 0,
            "missing_target_families_remain_missing": True,
            "houses": rows, "manifest_house_count": len(expected_houses),
            "no_recollection": True,
            "audit_rule": "FAMILY requires content_root and complete histories/traces; partial FAMILY.json is not accepted"}


def proposal_index(repair_run: Path) -> dict[str, dict[str, Any]]:
    value = load_json_or_none(repair_run / "PROPOSALS_rqfALeAoiTq.json")
    if not isinstance(value, list):
        return {}
    return {str(row["id"]): row for row in value if isinstance(row, dict) and row.get("id")}


def v1_proposal_states(repair_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = proposal_index(repair_run)
    rows = parse_jsonl(repair_run / "COLLECTION_ATTEMPTS.jsonl")
    starts = {row.get("proposal"): row for row in rows if row.get("status") == "START" and row.get("proposal")}
    rejected = {row.get("proposal"): row for row in rows if row.get("status") == "REJECTED" and row.get("proposal")}
    completed: list[dict[str, Any]] = []
    in_progress: list[dict[str, Any]] = []
    not_reached: list[dict[str, Any]] = []
    root = repair_run / "proposals"
    for proposal_id, proposal in index.items():
        folder = root / proposal_id
        family = load_json_or_none(folder / "FAMILY.json") if folder.exists() else None
        failure = load_json_or_none(folder / "FAILURE.json") if folder.exists() else None
        if completed_family(family):
            state = "ACCEPTED_FAMILY"
        elif family is not None:
            state = "PARTIAL_FAMILY"
        elif proposal_id in rejected or failure is not None:
            state = "COMPLETED_REJECTION"
        elif proposal_id in starts:
            state = "IN_PROGRESS"
        else:
            state = "NOT_REACHED"
        row = {"proposal": proposal_id, "state": state,
               "failure_path": rel(folder / "FAILURE.json") if failure is not None else None,
               "family_path": rel(folder / "FAMILY.json") if family is not None else None,
               "start_line": starts.get(proposal_id, {}).get("_line"),
               "rejection_line": rejected.get(proposal_id, {}).get("_line"),
               "proposal_source": proposal.get("source_proposal_path")}
        if state in {"COMPLETED_REJECTION", "ACCEPTED_FAMILY"}:
            completed.append(row)
        elif state in {"IN_PROGRESS", "PARTIAL_FAMILY"}:
            in_progress.append(row)
        else:
            not_reached.append(row)
    return completed, in_progress, not_reached


def neutral_start_audit(repair_run: Path, house: dict[str, Any]) -> dict[str, Any]:
    index = proposal_index(repair_run)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    root = repair_run / "proposals"
    for proposal_id, proposal in sorted(index.items()):
        folder = root / proposal_id
        if not folder.is_dir():
            continue
        yaw_rows = yaw_search_rows(folder)
        for yaw_dir in sorted(folder.glob("yaw_*")):
            if not yaw_dir.is_dir():
                continue
            match = re.fullmatch(r"yaw_(\d+)", yaw_dir.name)
            if not match:
                continue
            yaw = int(match.group(1))
            probe = yaw_dir / "INITIAL_VIEW_PROBE.json"
            entry: dict[str, Any] = {"hub": proposal.get("hub"), "proposal": proposal_id,
                                     "anchor_A": proposal.get("role_indices", [None, None])[0],
                                     "anchor_B": proposal.get("role_indices", [None, None])[1],
                                     "yaw": yaw, "yaw_key": f"{proposal.get('hub')}:{proposal.get('role_indices')}:{yaw}",
                                     "evidence_path": rel(probe) if probe.exists() else None,
                                     "evidence_sha256": sha256_file(probe) if probe.exists() else None,
                                     "actual_probe_frames": None, "roles_triggered": [],
                                     "adjacent_pixel_evidence": [], "original_checker_result": original_error_for_yaw(yaw_rows, f"{yaw:02d}")}
            if not probe.exists():
                entry.update({"classification": "EVIDENCE_MISSING", "checker_result": "NOT_REACHED"})
                counts["EVIDENCE_MISSING"] += 1
                rows.append(entry)
                continue
            trace = load_json_or_none(probe)
            try:
                compiler = make_compiler(proposal, house)
                observations = trace["observations"]
                events, hits = hit_details(observations, compiler)
                roles = sorted({hit["role"] for hit in hits})
                entry.update({"actual_probe_frames": len(observations), "roles_triggered": roles,
                              "adjacent_pixel_evidence": hits,
                              "checker_result": "ANCHOR_SEEN" if roles else "NO_ANCHOR_NEUTRAL_START_VIEW"})
                # This is per yaw.  Eight-yaw conclusions are computed below;
                # absence of a file is never treated as an invalid yaw.
                entry["classification"] = "INVALID_YAW" if roles else "VALID_NEUTRAL_START"
                counts[entry["classification"]] += 1
            except Exception as exc:
                entry.update({"classification": "ERROR", "checker_result": "ERROR",
                              "diagnostic_error": repr(exc), "traceback": traceback.format_exc(limit=3)})
                counts["ERROR"] += 1
            rows.append(entry)
    by_proposal: list[dict[str, Any]] = []
    for proposal_id in sorted(index):
        ours = [row for row in rows if row["proposal"] == proposal_id]
        valid = [row for row in ours if row.get("classification") == "VALID_NEUTRAL_START"]
        missing = [row for row in ours if row.get("classification") == "EVIDENCE_MISSING"]
        if not ours:
            classification = "EVIDENCE_MISSING"
        elif len(ours) < len(YAW_SET):
            classification = "EVIDENCE_MISSING"
        elif len(valid) == 0:
            classification = "ALL_EIGHT_INVALID"
        elif len(valid) == len(YAW_SET):
            classification = "ALL_EIGHT_NEUTRAL_START_VALID"
        else:
            classification = "SOME_YAWS_INVALID"
        by_proposal.append({"proposal": proposal_id, "measured_yaws": len(ours),
                            "valid_neutral_yaws": [x["yaw"] for x in valid],
                            "missing_yaws": [x["yaw"] for x in missing],
                            "proposal_classification": classification})
    return {"version": VERSION, "house": HOUSE, "split": "TEST", "see2_pixels": SEE2_PIXELS,
            "key": "(hub, anchor_A, anchor_B, yaw)", "yaw_candidates": YAW_SET,
            "yaw_records": rows, "proposal_rollup": by_proposal,
            "counts": dict(counts),
            "interpretation": "per-yaw invalidity, all-eight invalidity, and missing evidence are separate labels; no universal house/orientation claim is made"}


def isolation_audit(repair_run: Path, house: dict[str, Any]) -> dict[str, Any]:
    index = proposal_index(repair_run)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for proposal_id, proposal in sorted(index.items()):
        folder = repair_run / "proposals" / proposal_id
        if not folder.is_dir():
            continue
        yaw_error = {f"{int(row.get('yaw')):02d}": row for row in yaw_search_rows(folder)
                     if str(row.get("yaw", "")).lstrip("-").isdigit()}
        for path in sorted(folder.glob("yaw_*/H_*_DISCOVERY.json")):
            history_id = path.name[:-len("_DISCOVERY.json")]
            yaw = path.parent.name.removeprefix("yaw_")
            error = None
            row = yaw_error.get(yaw)
            if isinstance(row, dict):
                error = row.get("error", row.get("reason", row.get("status")))
                if error is not None and not isinstance(error, str):
                    error = repr(error)
            result = trace_summary(path, proposal, house, history_id, error)
            counts[str(result.get("offline_recompute"))] += 1
            rows.append(result)
        # A partial FAMILY is evidence of the publication defect, not a valid
        # accepted family; make it visible in the audit without publishing it.
        family = load_json_or_none(folder / "FAMILY.json")
        if family is not None and not completed_family(family):
            rows.append({"proposal": proposal_id, "yaw": None, "history_id": None,
                         "evidence_path": rel(folder / "FAMILY.json"),
                         "evidence_sha256": sha256_file(folder / "FAMILY.json"),
                         "offline_recompute": "PARTIAL_FAMILY_NOT_ACCEPTED",
                         "first_contamination_stage": "NOT_APPLICABLE",
                         "stage_boundary_status": "NOT_APPLICABLE",
                         "original_error": "duplicate exclusive FAMILY publication left content_root absent"})
            counts["PARTIAL_FAMILY_NOT_ACCEPTED"] += 1
    return {"version": VERSION, "house": HOUSE, "split": "TEST", "see2_pixels": SEE2_PIXELS,
            "history_cap": HISTORY_CAP, "rows": rows, "counts": dict(counts),
            "stage_vocabulary": ["outbound_turn", "outbound", "witness_search", "perturbation",
                                  "return", "public_tail", "UNRESOLVED"],
            "stage_rule": "UNRESOLVED when old trace contains no explicit phase boundary; no route-shape inference"}


def failure_funnel(repair_run: Path, neutral: dict[str, Any], isolation: dict[str, Any]) -> dict[str, Any]:
    rows = parse_jsonl(repair_run / "COLLECTION_ATTEMPTS.jsonl")
    proposal_states, in_progress, not_reached = v1_proposal_states(repair_run)
    yaw_counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    for proposal_id in {row.get("proposal") for row in rows if row.get("proposal")}:
        folder = repair_run / "proposals" / str(proposal_id)
        for row in yaw_search_rows(folder):
            value = row.get("error") or row.get("reason") or row.get("status") or "NOT_REACHED"
            value = value if isinstance(value, str) else repr(value)
            yaw_counts[value] += 1
            if value.startswith("Rejected(") or "REJECTED" in value:
                categories["PHYSICAL_REJECTED"] += 1
            elif value in {"CERTIFIED", "ACCEPTED", "PASS"}:
                categories["CERTIFIED_OR_ACCEPTED"] += 1
            elif value == "NOT_REACHED":
                categories["NOT_REACHED"] += 1
            else:
                categories["ERROR"] += 1
            if "NEUTRAL" in value:
                phase_counts["neutral_start"] += 1
            elif "SEE2" in value or "WITNESS" in value:
                phase_counts["witness_search"] += 1
            else:
                phase_counts["UNRESOLVED"] += 1
    for row in isolation.get("rows", []):
        if row.get("original_error") and row.get("offline_recompute"):
            phase_counts[str(row.get("first_contamination_stage", "UNRESOLVED"))] += 1
    return {"version": VERSION, "house": HOUSE, "split": "TEST",
            "proposal_denominator": {"v1_indexed": len(proposal_states) + len(in_progress) + len(not_reached),
                                     "v1_completed_rejections_or_accepted": len(proposal_states),
                                     "v1_partial_or_in_progress": len(in_progress),
                                     "v1_not_reached": len(not_reached),
                                     "old_formal_attempts": 117},
            "yaw_denominator": {"rows_by_original_result": dict(yaw_counts),
                                 "classification": dict(categories)},
            "stage_denominator": dict(phase_counts),
            "physical_rejection_definition": "only explicit Rejected result; unexpected exception is ERROR",
            "evidence_missing_and_not_reached_are_not_physical_passes": True,
            "partial_family_count": sum(1 for row in proposal_states + in_progress if row.get("state") == "PARTIAL_FAMILY"),
            "accepted_new_families": 0,
            "new_physical_candidate_executions": 0}


def source_lock(old_run: Path, repair_run: Path) -> dict[str, Any]:
    names = ["isolation_diagnosis_v2.py", "test_isolation_diagnosis_v2.py", "collect.py",
             "collect_repair_v1.py", "repair_resume_v1.py", "evaluator_v16.py", "v16_common.py",
             "DATA_MANIFEST.json", "PROTOCOL.json"]
    paths = [HERE / name for name in names] + [
        LINE / "data_pipeline" / "mechanism_factory_v2" / "compiler.py",
        old_run / "SOURCE_LOCK.json", repair_run / "SOURCE_LOCK.json",
        old_run / "PROTOCOL.json", repair_run / "PROTOCOL.json"]
    files = {}
    for path in paths:
        if path.is_file():
            files[rel(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"version": VERSION, "run_id": RUN_ID, "baseline_commit_reviewed": BASELINE_COMMIT,
            "entrypoint": rel(HERE / "isolation_diagnosis_v2.py"),
            "actual_dependencies": sorted(files), "files": files,
            "source_isolation": "independent v2 entrypoint; v1 common modules are read-only inputs; no v1 worktree switch",
            "runtime": {"python_stdlib": True, "simulator": False, "gpu": False, "qwen": False,
                        "training": False, "pipeline_all": False}}


def decision() -> dict[str, Any]:
    return {"decision": "B", "decision_date": "2026-09-20",
            "approved_version": VERSION, "approved_run_id": RUN_ID,
            "suggested_service": "q35n-v16-repair-rqf-v2-diag-20260920-01.service",
            "scope": "offline diagnosis + CPU acceptance only",
            "authorization": {"new_physical_candidate_executions": 0,
                               "reuse_certified_families_by_value": 24,
                               "modify_v1": False, "stop_or_pause_v1": False,
                               "load_qwen": False, "enter_G1": False,
                               "train": False, "evaluate_navigation": False,
                               "pipeline_all": False, "automatic_promotion": False},
            "immutable_requirements": {"SEE2_pixels_each_adjacent_frame": SEE2_PIXELS,
                                        "collision": "any collision rejects",
                                        "active_STOP": True, "full_decision_cap": FULL_DECISION_CAP,
                                        "history_decision_cap": HISTORY_CAP,
                                        "house": HOUSE, "split": "TEST", "missing_families": 2},
            "terminal_statuses": ["DIAGNOSIS_ONLY_COMPLETE", "BLOCKED_EVIDENCE_MISSING",
                                   "BLOCKED_CPU_TEST", "BLOCKED_RESOURCE"]}


def patch_plan() -> dict[str, Any]:
    return {"version": VERSION, "status": "PLAN_ONLY_NO_PHYSICAL_SEARCH",
            "findings": [
                {"id": "F1", "interface": "collect_repair_v1._attempt_yaw / collect_family_repair",
                 "change": "assemble content_root and all certificate fields before a single exclusive FAMILY.json publication; recovery validates content_root and certificates; never overwrite", "verification": "one-time publish and partial recovery CPU tests"},
                {"id": "F2", "interface": "collector exception boundary",
                 "change": "catch explicit Rejected as physical rejection; unexpected Exception becomes ERROR with traceback; evidence absence is EVIDENCE_MISSING and unchecked checks are NOT_REACHED", "verification": "classification CPU tests and funnel category separation"},
                {"id": "F3", "interface": "runtime entrypoint",
                 "change": "use a diagnosis-only entrypoint; do not call repair_resume_v1, promote_pilot, or pipeline --phase all", "verification": "static guard and receipt"},
            ],
            "deterministic_candidate_order_for_future_authorized_validation": {
                "order": "hub ascending, proposal id lexicographic, yaw in [0,3,...,21], then declared target order",
                "candidate_denominator": "only explicitly enumerated candidates in the sealed future validation manifest",
                "budget": "not authorized or executed in this run; propose a small CPU-rechecked diagnostic sample before any physical execution",
                "selection_by_model_score": False},
            "unchanged_rules": ["SEE2 same instance and 256 pixels in both adjacent actual frames",
                                 "collision rejection", "active STOP and terminal witness", "500 full decisions / 240 history decisions", "information isolation"]}


def report(snapshot: dict[str, Any], reuse: dict[str, Any], funnel: dict[str, Any],
           neutral: dict[str, Any], isolation: dict[str, Any], cpu: dict[str, Any]) -> str:
    counts = isolation.get("counts", {})
    neutral_counts = neutral.get("counts", {})
    return f"""# Q35N V16 rqf v2 离线诊断报告

## 结论与边界

本 run 只执行了 `{VERSION}` 的封存输入、CPU 证据重算和正确性验收。新增物理候选执行数为 **0**；没有启动 Habitat、GPU、Qwen、训练、导航评测、G1 或自动晋级。允许的结论仍是：**修复候选仍未找到两个物理合法族。**这不是训练失败、模型失败或导航收益结论。

输入截止时间：`{snapshot['snapshot_cutoff_iso']}`。v1 失败证据按文件哈希封存；活跃目录未 chmod、移动、改名或加锁。旧 run、旧源码锁和失败结果未覆盖。

## 复用与缺口

- `v16_formal_001` 逐 HOUSE 校验得到 `{reuse['validated_complete_family_count']}` 个完整族（批准口径为 24），按值复用，不重新采集。
- `{HOUSE}`/TEST 仍缺 2 个族；v1 新修复 run 的新增接受数为 0。
- v1 的一个 `FAMILY.json` 被识别为 `PARTIAL_FAMILY`：缺 `content_root`，不可接纳；这与成功路径重复排他发布的静态缺陷相符，但不把它倒推为所有物理拒绝的原因。

## 初始中性视角（问题 A）

审计键为 `(hub, anchor_A, anchor_B, yaw)`，每个已保存探测逐帧记录 role、实例和相邻帧像素。逐 yaw 的计数为：`{json.dumps(neutral_counts, ensure_ascii=False, sort_keys=True)}`。proposal 汇总严格区分某些 yaw 不合法、八个 yaw 均不合法和证据缺失；没有把未保存的证据写成“八个均不合法”，也没有外推为整屋所有朝向均不合法。

## 完整历史隔离（问题 B）

逐条读取已保存 `H_*_DISCOVERY.json`，继续调用原 `Compiler.atoms()`：同一 eligible 实例在相邻两帧各至少 256 像素才构成 SEE2，不拼接尝试、不跨实例合并、不以房间替代 role。重算计数：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`。

旧 trace 没有可靠的阶段边界字段，因此首次污染阶段按要求记为 `UNRESOLVED`，没有按路线外观猜测“去程/返程/公共尾段”。证据缺失、未到达和程序错误均与物理拒绝分开。

## 修复与 CPU 验收

`GENERATOR_PATCH_PLAN.json` 只提出局部修补：完整组装后一次发布、异常分流、独立诊断入口；本次不执行物理验证搜索。`CPU_TEST_RESULT.json` 记录一次性发布、半成品恢复拒绝、SEE2 同实例阈值、旧文件不变和禁止模型阶段的测试结果：`{cpu.get('status')}`。

## 下一步门槛

只有主 agent 审核本诊断和有限验证方案后，才可另行授权物理验证。候选顺序、候选分母和预算在 `GENERATOR_PATCH_PLAN.json` 中明确；本 run 正常终止状态为 `DIAGNOSIS_ONLY_COMPLETE`，不自动开搜。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only V16 rqf isolation diagnosis")
    parser.add_argument("--source-certified-run", default=str(RUNS / "v16_formal_001"))
    parser.add_argument("--source-failure-run", default=str(RUNS / "v16_repair_rqf_001"))
    parser.add_argument("--run", default=str(RUNS / RUN_ID))
    parser.add_argument("--cutoff-unix", type=float, default=None)
    parser.add_argument("--cpu-result", default=None,
                        help="JSON result produced by the independent CPU test runner")
    args = parser.parse_args(argv)
    guard = diagnosis_only_guard()
    run = Path(args.run).resolve()
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"V2_RUN_EXISTS_REFUSE_OVERWRITE:{run}")
    run.mkdir(parents=True, exist_ok=False)
    with (run / "RUN.lock").open("x", encoding="utf-8") as stream:
        stream.write("diagnosis-only\n")
        stream.flush()
        os.fsync(stream.fileno())
    old_run, repair_run = Path(args.source_certified_run).resolve(), Path(args.source_failure_run).resolve()
    start = time.time()
    cutoff = float(args.cutoff_unix if args.cutoff_unix is not None else start)
    manifest = load_json(HERE / "DATA_MANIFEST.json")
    house = next(row for row in manifest["houses"] if row.get("house") == HOUSE)
    spec = decision()
    spec.update({"entrypoint": rel(HERE / "isolation_diagnosis_v2.py"),
                 "run_path": rel(run), "source_certified_run": rel(old_run),
                 "source_failure_run": rel(repair_run), "snapshot_cutoff_unix": cutoff,
                 "new_physical_candidate_executions": 0})
    write_once(run / "PRO_AGENT_DECISION.json", spec)
    write_once(run / "REPAIR_SPEC.json", {"version": VERSION, "run_id": RUN_ID,
                                           "phase": "offline_diagnosis_cpu_acceptance",
                                           "house": HOUSE, "split": "TEST", "target_missing_families": 2,
                                           "new_physical_candidate_executions": 0,
                                           "diagnosis_only": True, "guard": guard})
    snapshot = input_snapshot(old_run, repair_run, cutoff)
    write_once(run / "INPUT_SNAPSHOT.json", snapshot)
    lock = source_lock(old_run, repair_run)
    write_once(run / "SOURCE_LOCK.json", lock)
    reuse = reuse_audit(old_run, manifest)
    write_once(run / "REUSE_AUDIT.json", reuse)
    neutral = neutral_start_audit(repair_run, house)
    write_once(run / "NEUTRAL_START_AUDIT.json", neutral)
    isolation = isolation_audit(repair_run, house)
    write_once(run / "ISOLATION_AUDIT.json", isolation)
    funnel = failure_funnel(repair_run, neutral, isolation)
    write_once(run / "FAILURE_FUNNEL.json", funnel)
    plan = patch_plan()
    write_once(run / "GENERATOR_PATCH_PLAN.json", plan)
    if args.cpu_result:
        cpu_result = load_json(Path(args.cpu_result))
    else:
        cpu_result = {"status": "BLOCKED_CPU_TEST", "reason": "independent CPU result was not supplied",
                      "test_entrypoint": rel(HERE / "test_isolation_diagnosis_v2.py"),
                      "new_physical_candidate_executions": 0}
    cpu_result.setdefault("test_entrypoint", rel(HERE / "test_isolation_diagnosis_v2.py"))
    cpu_result.setdefault("new_physical_candidate_executions", 0)
    write_once(run / "CPU_TEST_RESULT.json", cpu_result)
    terminal_status = "DIAGNOSIS_ONLY_COMPLETE" if cpu_result.get("status") == "PASS" else "BLOCKED_CPU_TEST"
    receipt = {"version": VERSION, "run_id": RUN_ID, "service": "q35n-v16-repair-rqf-v2-diag-20260920-01.service",
               "phase": "offline_diagnosis_cpu_acceptance", "start_unix": start,
               "diagnosis_unix": time.time(), "gpu_used": False, "habitat_used": False,
               "qwen_loaded": False, "training_started": False, "navigation_evaluation_started": False,
               "new_physical_candidate_executions": 0, "automatic_promotion": False,
               "v1_service_action": "none", "terminal_status": terminal_status}
    write_once(run / "RESOURCE_AND_STAGE_RECEIPT.json", receipt)
    write_text_once(run / "DIAGNOSIS_REPORT_ZH.md", report(snapshot, reuse, funnel, neutral, isolation, cpu_result))
    write_once(run / "STATUS.json", {"status": terminal_status, "run_id": RUN_ID,
                                     "version": VERSION, "new_physical_candidate_executions": 0,
                                     "accepted_new_families": 0})
    print(json.dumps({"run": str(run), "status": terminal_status,
                      "isolation_rows": len(isolation.get("rows", [])),
                      "reuse_families": reuse.get("validated_complete_family_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
