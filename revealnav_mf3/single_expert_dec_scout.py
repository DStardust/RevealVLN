"""MF3ZP single-expert DEC calibration scout.

This module prepares outcome-blind selections and blinded review material,
seals reproducibility metadata, and scores completed first/retest reviews.  It
does not call Qwen, load navigation outcomes, or authorize downstream work.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence

from .evidence_constraints import InstructionEvidenceGraph
from .evidence_uad import derive_constraint_uad
from .human_dec_schema import DEC_ROLES, DecRole, validate_review_row
from .qwen_evidence_annotation import parse_instruction_response, stable_sha256


ROOT = Path(__file__).resolve().parents[1]
BASE_REVIEW_COMMIT = "3e16465d095e4e8ae36ad4ea310f6e02fc9737b1"
REVISION = "mf3zp_single_expert_dec_scout_v1"
OUTPUT = ROOT / "artifacts/training/mf3zp_single_expert_dec_scout_v1"

FORMAL_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_PROTOCOL.json"
CORRECTNESS_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_V1_1_CORRECTNESS_PROTOCOL.json"
PILOT_EVENTS = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEAL_EVENTS.jsonl"
PILOT_SELECTION = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEAL_PILOT_SELECTION.json"
QWEN_STATUS = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_EVIDENCE_V1_1_STATUS.json"
QWEN_ROOT = ROOT / "artifacts/training/mf3zp_revealskill_v1/qwen_preannotations"
INSTRUCTION_DIR = QWEN_ROOT / "instruction"
EVIDENCE_DIR = QWEN_ROOT / "evidence"
EVIDENCE_V11_DIR = QWEN_ROOT / "evidence_v1_1"
SOURCE_REQUESTS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_ANNOTATION_REQUESTS.jsonl"

CLOSURE_PROTOCOL = OUTPUT / "MF3ZP_REPRODUCIBILITY_CLOSURE_PROTOCOL.json"
SCOUT_PROTOCOL = OUTPUT / "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_PROTOCOL.json"
SCOUT_SELECTION = OUTPUT / "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_SELECTION.json"
RETEST_SELECTION = OUTPUT / "MF3ZP_SINGLE_EXPERT_RETEST_SELECTION.json"
REVIEW_TEMPLATE = OUTPUT / "MF3ZP_SINGLE_EXPERT_REVIEW_TEMPLATE.jsonl"
REVIEW_HTML = OUTPUT / "MF3ZP_SINGLE_EXPERT_REVIEW.html"
STATUS_PATH = OUTPUT / "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_STATUS.json"
FIRST_VALIDATION = OUTPUT / "MF3ZP_SINGLE_EXPERT_FIRST_REVIEW_VALIDATION.json"
RETEST_TEMPLATE = OUTPUT / "MF3ZP_SINGLE_EXPERT_RETEST_TEMPLATE.jsonl"
RETEST_HTML = OUTPUT / "MF3ZP_SINGLE_EXPERT_RETEST.html"
RESULT_PATH = OUTPUT / "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_RESULT.json"

SELECTION_SALT = "mf3zp-single-expert-dec-scout-v1:"
RETEST_SALT = "mf3zp-single-expert-retest-v1:"
RETEST_ORDER_SALT = "mf3zp-single-expert-retest-order-v1:"
EVENTS_PER_DOMAIN = 40
RETEST_PER_DOMAIN = 10
REVIEW_WINDOW = 5
UAD_K = 3
MIN_QWEN_D_SUPPORT = 20

EXPECTED_HISTORICAL_SHA256 = {
    str(FORMAL_PROTOCOL.relative_to(ROOT)): "d0f09395b86804d3afc58f4ec946afc7dfaffd1637c7b8a66a776d58a17cc0c9",
    str(CORRECTNESS_PROTOCOL.relative_to(ROOT)): "670b444d0c5920e084629c99979b9804026dbb47759c2d737248209590fca9ba",
    str(PILOT_EVENTS.relative_to(ROOT)): "5636fa6991287aa61a4124df292e65e044cb6f968e9f6a24573cf417623a69d2",
    str(PILOT_SELECTION.relative_to(ROOT)): "a93351616ebf21ef11d8bfbc49efa2e5d2a5e7d777abf81491b8c4cdab0d9dd8",
    str(QWEN_STATUS.relative_to(ROOT)): "a824445dc73521ac340efb90f648ecd22f7360594458ebdd8cb05abd213d3f02",
}

IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZP_SINGLE_EXPERT_DEC_SCOUT.md",
    "METHOD_REVISION_3ZP_REPRODUCIBILITY_CLOSURE.md",
    "METHOD_REVISION_3ZP_HUMAN_REVIEW_CORRECTNESS.md",
    "revealnav_mf3/human_dec_schema.py",
    "revealnav_mf3/single_expert_dec_scout.py",
    "scripts/seal_mf3zp_single_expert_dec_scout.py",
    "scripts/prepare_mf3zp_single_expert_review.py",
    "scripts/prepare_mf3zp_single_expert_retest.py",
    "scripts/audit_mf3zp_single_expert_review.py",
    "scripts/audit_mf3zp_labels_v1_2.py",
    "scripts/verify_mf3zp_reproducibility_closure.py",
    "tests/test_mf3zp_single_expert_selection.py",
    "tests/test_mf3zp_single_expert_blinding.py",
    "tests/test_mf3zp_single_expert_retest.py",
    "tests/test_mf3zp_dec_scoring.py",
    "tests/test_mf3zp_false_decisive.py",
    "tests/test_mf3zp_reproducibility_closure.py",
    "tests/test_mf3zp_future_multi_review_adjudication.py",
)

PUBLIC_CLOSED = {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}


class ScoutError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if not path.is_file() or path.is_symlink() or root not in resolved.parents:
        raise ScoutError(f"invalid project-local file: {path}")
    return {
        "path": str(resolved.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_records(paths: Iterable[Path]) -> dict[str, object]:
    records = [inventory(path) for path in sorted(paths)]
    return {
        "count": len(records),
        "inventory_sha256": stable_sha256(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
    }


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ScoutError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ScoutError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScoutError(f"cannot read JSONL: {path}") from error
    for line_no, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise ScoutError(f"invalid JSONL: {path}:{line_no}") from error
        if not isinstance(value, dict):
            raise ScoutError(f"JSON object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise ScoutError(f"empty JSONL: {path}")
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ScoutError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ScoutError(f"stale partial: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def _hash_key(salt: str, event_id: object) -> str:
    return hashlib.sha256((salt + str(event_id)).encode("utf-8")).hexdigest()


def _round_robin(
    events: Sequence[Mapping[str, object]],
    *,
    count: int,
    salt: str,
) -> list[dict[str, object]]:
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in events:
        by_scene[str(event["scene_id"])].append(event)
    for values in by_scene.values():
        values.sort(key=lambda event: (_hash_key(salt, event["event_id"]), str(event["event_id"])))
    selected: list[dict[str, object]] = []
    index = 0
    scene_ids = sorted(by_scene)
    while len(selected) < count:
        added = False
        for scene_id in scene_ids:
            if index < len(by_scene[scene_id]):
                selected.append(dict(by_scene[scene_id][index]))
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ScoutError("not enough events for deterministic scene round-robin")
        index += 1
    return selected


def select_scout_events(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(events) != 300 or Counter(str(row["dataset"]) for row in events) != {"R2R": 150, "RxR": 150}:
        raise ScoutError("frozen 300-event population drift")
    selected: list[dict[str, object]] = []
    for domain in ("R2R", "RxR"):
        domain_rows = [row for row in events if row["dataset"] == domain]
        selected.extend(
            _round_robin(domain_rows, count=EVENTS_PER_DOMAIN, salt=SELECTION_SALT)
        )
    if len({str(row["event_id"]) for row in selected}) != 80:
        raise ScoutError("scout selection contains duplicate event IDs")
    return selected


def select_retest_events(selected: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    retest: list[dict[str, object]] = []
    for domain in ("R2R", "RxR"):
        rows = [row for row in selected if row["dataset"] == domain]
        retest.extend(_round_robin(rows, count=RETEST_PER_DOMAIN, salt=RETEST_SALT))
    retest.sort(key=lambda row: (_hash_key(RETEST_ORDER_SALT, row["event_id"]), str(row["event_id"])))
    if len(retest) != 20 or Counter(str(row["dataset"]) for row in retest) != {"R2R": 10, "RxR": 10}:
        raise ScoutError("retest selection balance drift")
    return retest


def _instruction_key(instruction: str) -> str:
    return stable_sha256({"instruction": instruction.strip()})


def load_graph(instruction: str) -> InstructionEvidenceGraph:
    record = read_json(INSTRUCTION_DIR / f"{_instruction_key(instruction)}.json")
    if record.get("human_verified") is not False or record.get("gold") is not False:
        raise ScoutError("Qwen graph provenance drift")
    graph = parse_instruction_response(record["response"], instruction=instruction)
    if graph.canonical_sha256() != record.get("constraint_graph_sha256"):
        raise ScoutError("Qwen graph hash drift")
    return graph


def _selection_summary(event: Mapping[str, object]) -> dict[str, object]:
    graph = load_graph(str(event["instruction"]))
    instruction = str(event["instruction"])
    return {
        "event_id": str(event["event_id"]),
        "dataset": str(event["dataset"]),
        "scene_id": str(event["scene_id"]),
        "trigger_types": list(event["trigger_types"]),
        "prefix_depth": int(event["prefix_end"]) - int(event["prefix_start"]) + 1,
        "instruction_length_chars": len(instruction.strip()),
        "instruction_word_count": len(instruction.split()),
        "qwen_constraint_count": len(graph.constraints),
    }


def build_selection_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    events = read_jsonl(PILOT_EVENTS)
    selected = select_scout_events(events)
    retest = select_retest_events(selected)
    summaries = [_selection_summary(row) for row in selected]
    retest_summaries = [_selection_summary(row) for row in retest]
    domains = Counter(str(row["dataset"]) for row in summaries)
    triggers = Counter(trigger for row in summaries for trigger in row["trigger_types"])
    selection = {
        "schema_version": "revealnav-mf3zp-single-expert-dec-selection/1",
        "revision": REVISION,
        "status": "SEALED_OUTCOME_AND_QWEN_FACTOR_BLIND_SELECTION",
        "selection_rule": {
            "salt": SELECTION_SALT,
            "within_scene": "SHA256(salt + event_id)",
            "scene_order": "lexicographic_raw_scene_id",
            "allocation": "scene_round_robin_per_domain",
            "count_per_domain": EVENTS_PER_DOMAIN,
        },
        "event_count": len(summaries),
        "domain_counts": dict(sorted(domains.items())),
        "raw_scene_count": len({row["scene_id"] for row in summaries}),
        "trigger_counts": dict(sorted(triggers.items())),
        "event_ids_sha256": stable_sha256([row["event_id"] for row in summaries]),
        "events": summaries,
        "selection_inputs": ["dataset", "scene_id", "event_id"],
        "qwen_factor_labels_read": False,
        "outcome_payload_read": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    retest_value = {
        "schema_version": "revealnav-mf3zp-single-expert-retest-selection/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_FIRST_EXPERT_REVIEW",
        "selection_salt": RETEST_SALT,
        "presentation_order_salt": RETEST_ORDER_SALT,
        "event_count": len(retest_summaries),
        "domain_counts": dict(sorted(Counter(row["dataset"] for row in retest_summaries).items())),
        "raw_scene_count": len({row["scene_id"] for row in retest_summaries}),
        "event_ids_in_blind_order": [row["event_id"] for row in retest_summaries],
        "event_ids_sha256": stable_sha256([row["event_id"] for row in retest_summaries]),
        "events": retest_summaries,
        "first_pass_labels_read": False,
        "qwen_factor_labels_read": False,
        "outcome_payload_read": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    return selection, retest_value


def write_selection_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    selection, retest = build_selection_artifacts()
    atomic_json(SCOUT_SELECTION, selection)
    atomic_json(RETEST_SELECTION, retest)
    return selection, retest


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _base_is_ancestor() -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_REVIEW_COMMIT, "HEAD"],
        cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _verify_historical_files() -> None:
    for relative, expected in EXPECTED_HISTORICAL_SHA256.items():
        path = ROOT / relative
        if sha256_file(path) != expected:
            raise ScoutError(f"historical MF3ZP file drift: {relative}")
    formal = read_json(FORMAL_PROTOCOL)
    correction = read_json(CORRECTNESS_PROTOCOL)
    if formal.get("public_split_access") != PUBLIC_CLOSED or correction.get("public_split_access") != PUBLIC_CLOSED:
        raise ScoutError("historical public split boundary drift")
    authorization = formal.get("authorization", {})
    for key in (
        "oracle_headroom", "ree_training", "skill_rollout_collection",
        "skill_policy_training", "checkpoint_generation", "public_evaluation",
    ):
        if authorization.get(key) is not False:
            raise ScoutError(f"historical downstream authorization opened: {key}")
    status = read_json(QWEN_STATUS)
    if status.get("valid") != 538 or status.get("human_verified") is not False or status.get("gold") is not False:
        raise ScoutError("Qwen provisional status drift")


def _record_inventories() -> dict[str, object]:
    instruction = inventory_records(INSTRUCTION_DIR.glob("*.json"))
    evidence_paths = list(EVIDENCE_DIR.glob("*.json")) + list(EVIDENCE_V11_DIR.glob("*.json"))
    evidence = inventory_records(evidence_paths)
    if instruction["count"] != 141 or evidence["count"] != 538:
        raise ScoutError("Qwen record population drift")
    request_ids = [path.stem for path in evidence_paths]
    if len(set(request_ids)) != 538:
        raise ScoutError("Qwen evidence request identity collision")
    return {"instruction_graph_records": instruction, "qwen_evidence_records": evidence}


def build_closure_protocol() -> dict[str, object]:
    _verify_historical_files()
    if not _base_is_ancestor():
        raise ScoutError("reviewed base commit is not an ancestor of HEAD")
    if not SCOUT_SELECTION.is_file() or not RETEST_SELECTION.is_file():
        raise ScoutError("selection artifacts must be sealed before closure")
    implementation = {name: inventory(ROOT / name) for name in IMPLEMENTATION_FILES}
    return {
        "schema_version": "revealnav-mf3zp-reproducibility-closure/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_SINGLE_EXPERT_LABELS",
        "base_review_commit": BASE_REVIEW_COMMIT,
        "head_at_seal": _git_head(),
        "base_is_current_history_ancestor": True,
        "historical_files": {
            relative: inventory(ROOT / relative) for relative in EXPECTED_HISTORICAL_SHA256
        },
        "record_inventories": _record_inventories(),
        "selection_files": {
            "scout": inventory(SCOUT_SELECTION),
            "retest": inventory(RETEST_SELECTION),
        },
        "scout_implementation_inventory": implementation,
        "historical_protocol_source_commit_rewritten": False,
        "verification_rule": "base ancestor plus byte-exact inventoried files; HEAD equality is not required",
        "result_write_policy": "refuse_existing",
        "qwen_api_calls": 0,
        "public_split_access": PUBLIC_CLOSED,
    }


def build_scout_protocol(closure: Mapping[str, object]) -> dict[str, object]:
    selection = read_json(SCOUT_SELECTION)
    retest = read_json(RETEST_SELECTION)
    if selection.get("domain_counts") != {"R2R": 40, "RxR": 40}:
        raise ScoutError("scout selection domain balance drift")
    if retest.get("domain_counts") != {"R2R": 10, "RxR": 10}:
        raise ScoutError("retest selection domain balance drift")
    return {
        "schema_version": "revealnav-mf3zp-single-expert-dec-scout-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_SINGLE_EXPERT_RESULTS",
        "base_review_commit": BASE_REVIEW_COMMIT,
        "reproducibility_closure": inventory(CLOSURE_PROTOCOL),
        "selection": inventory(SCOUT_SELECTION),
        "retest_selection": inventory(RETEST_SELECTION),
        "scientific_role": "annotation/data-readiness calibration; not a scientific gate",
        "qwen_model": "qwen3.8-max",
        "qwen_labels": "provisional",
        "selection_contract": {
            "events": 80, "R2R": 40, "RxR": 40,
            "salt": SELECTION_SALT, "retest_events": 20,
            "retest_R2R": 10, "retest_RxR": 10,
            "retest_salt": RETEST_SALT,
        },
        "review_contract": {
            "experts": 1,
            "blind_test_retest": True,
            "review_window_prefixes": REVIEW_WINDOW,
            "uad_stability_k": UAD_K,
            "qwen_proposed_dec_definition": "complete frozen event-agnostic Qwen constraint graph",
            "constraint_classification_roles": [
                "DEC_REQUIRED", "PREREQUISITE_ONLY", "FUTURE_NOT_RELEVANT",
                "REDUNDANT", "INCORRECT",
            ],
            "human_dec_membership_roles": [
                "DEC_REQUIRED", "PREREQUISITE_ONLY", "MISSING_DEC_CONSTRAINT",
            ],
            "retest_alignment": (
                "Qwen atoms by frozen ID; added atoms by exact normalized text plus "
                "manual mapping; membership mismatch is NOT_DEC disagreement"
            ),
            "qwen_factors_visible": False,
            "qwen_rationale_visible": False,
            "outcomes_visible": False,
            "future_frames_visible": False,
        },
        "readiness_thresholds": {
            "intra_uad_kappa_min": 0.75,
            "intra_e_kappa_min": 0.80,
            "dec_precision_min": 0.80,
            "dec_recall_min": 0.90,
            "qwen_expert_uad_accuracy_min": 0.80,
            "false_decisive_rate_max": 0.10,
            "minimum_qwen_d_support": MIN_QWEN_D_SUPPORT,
        },
        "forbidden_within_revision": [
            "qwen_model_change", "qwen_prompt_change", "qwen_recall",
            "event_replacement", "uad_k_change", "threshold_change",
            "dec_metric_change", "sge_semantics_change", "outcome_conditioned_dec",
        ],
        "formal_label_validity_pass": False,
        "authorization": {
            "oracle_headroom": False, "ree_training": False,
            "skill_rollout": False, "skill_policy_training": False,
            "checkpoint_generation": False, "public_evaluation": False,
        },
        "qwen_api_calls": 0,
        "human_labels_fabricated": False,
        "public_split_access": PUBLIC_CLOSED,
        "closure_snapshot": {
            "implementation_count": len(closure["scout_implementation_inventory"]),
            "instruction_inventory_sha256": closure["record_inventories"]["instruction_graph_records"]["inventory_sha256"],
            "evidence_inventory_sha256": closure["record_inventories"]["qwen_evidence_records"]["inventory_sha256"],
        },
    }


def seal_protocols() -> tuple[dict[str, object], dict[str, object]]:
    if CLOSURE_PROTOCOL.exists() or SCOUT_PROTOCOL.exists():
        raise ScoutError("refusing to overwrite a sealed scout protocol")
    closure = build_closure_protocol()
    atomic_json(CLOSURE_PROTOCOL, closure)
    try:
        scout = build_scout_protocol(closure)
        atomic_json(SCOUT_PROTOCOL, scout)
    except Exception:
        CLOSURE_PROTOCOL.unlink(missing_ok=True)
        raise
    return closure, scout


def verify_closure() -> dict[str, object]:
    closure = read_json(CLOSURE_PROTOCOL)
    if closure.get("base_review_commit") != BASE_REVIEW_COMMIT or not _base_is_ancestor():
        raise ScoutError("reviewed base commit ancestry failure")
    if closure.get("public_split_access") != PUBLIC_CLOSED or closure.get("qwen_api_calls") != 0:
        raise ScoutError("closure boundary drift")
    _verify_historical_files()
    for expected in closure.get("historical_files", {}).values():
        if inventory(ROOT / str(expected["path"])) != expected:
            raise ScoutError(f"historical inventory drift: {expected['path']}")
    current_records = _record_inventories()
    if current_records != closure.get("record_inventories"):
        raise ScoutError("Qwen record inventory drift")
    for expected in closure.get("selection_files", {}).values():
        if inventory(ROOT / str(expected["path"])) != expected:
            raise ScoutError(f"selection inventory drift: {expected['path']}")
    for expected in closure.get("scout_implementation_inventory", {}).values():
        if inventory(ROOT / str(expected["path"])) != expected:
            raise ScoutError(f"scout implementation drift: {expected['path']}")
    return closure


def verify_scout_protocol() -> dict[str, object]:
    verify_closure()
    protocol = read_json(SCOUT_PROTOCOL)
    if protocol.get("revision") != REVISION or protocol.get("status") != "SEALED_BEFORE_SINGLE_EXPERT_RESULTS":
        raise ScoutError("scout protocol identity drift")
    if protocol.get("public_split_access") != PUBLIC_CLOSED or protocol.get("qwen_api_calls") != 0:
        raise ScoutError("scout public/network boundary drift")
    if protocol.get("formal_label_validity_pass") is not False:
        raise ScoutError("single-expert scout cannot pass formal label validity")
    if any(value is not False for value in protocol.get("authorization", {}).values()):
        raise ScoutError("single-expert scout opened downstream authorization")
    for key, path in (("reproducibility_closure", CLOSURE_PROTOCOL), ("selection", SCOUT_SELECTION), ("retest_selection", RETEST_SELECTION)):
        if inventory(path) != protocol.get(key):
            raise ScoutError(f"scout protocol inventory drift: {key}")
    return protocol


def _task_index() -> dict[tuple[str, str, str, str, int], dict[str, object]]:
    result: dict[tuple[str, str, str, str, int], dict[str, object]] = {}
    for row in read_jsonl(SOURCE_REQUESTS):
        key = (
            str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]),
            str(row["event_id"]), int(row["prefix_step"]),
        )
        if key in result:
            raise ScoutError("duplicate causal prefix request")
        result[key] = row
    return result


def _event_by_id() -> dict[str, dict[str, object]]:
    events = read_jsonl(PILOT_EVENTS)
    result = {str(row["event_id"]): row for row in events}
    if len(result) != 300:
        raise ScoutError("frozen pilot event identity drift")
    return result


def _blank_factors(steps: Sequence[int]) -> list[dict[str, object]]:
    return [
        {"step": step, "instantiated": None, "distinguishable": None, "resolved": None}
        for step in steps
    ]


def build_review_rows(
    *,
    mode: str,
    selection: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    if mode not in {"first", "retest"}:
        raise ScoutError("unknown review mode")
    selection = selection or read_json(
        SCOUT_SELECTION if mode == "first" else RETEST_SELECTION
    )
    event_ids = (
        [str(row["event_id"]) for row in selection["events"]]
        if mode == "first" else [str(value) for value in selection["event_ids_in_blind_order"]]
    )
    events = _event_by_id()
    tasks = _task_index()
    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        event = events[event_id]
        graph = load_graph(str(event["instruction"]))
        decision = int(event["prefix_end"])
        start = max(int(event["prefix_start"]), decision - (REVIEW_WINDOW - 1))
        steps = list(range(start, decision + 1))
        prefix_rows = []
        decision_candidates: list[str] | None = None
        for step in steps:
            key = (
                str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]),
                str(event["source_observation_stream_id"]), step,
            )
            if key not in tasks:
                raise ScoutError(f"missing causal prefix task: {key}")
            task = tasks[key]
            candidates = [str(item["alias"]) for item in task["contract"]["current_candidates"]]
            if step == decision:
                decision_candidates = candidates
            prefix_rows.append({
                "step": step,
                "causal_storyboard_path": str(task["causal_storyboard"]["path"]),
                "current_panorama_path": str(task["current_panorama"]["path"]),
                "candidate_ids": candidates,
            })
        graph_rows = [constraint.as_mapping() for constraint in graph.constraints]
        row = {
            "schema_version": "revealnav-mf3zp-single-expert-dec-review/1",
            "review_mode": mode,
            "reviewer_id": "",
            "reviewer_blinded_to_outcomes": True,
            "reviewer_blinded_to_qwen_factors": True,
            "event_id": event_id,
            "dataset": str(event["dataset"]),
            "scene_id": str(event["scene_id"]),
            "episode_id": str(event["episode_id"]),
            "instruction": str(event["instruction"]),
            "decision_step": decision,
            "current_candidate_ids": decision_candidates or [],
            "review_prefix_start": start,
            "review_prefix_end": decision,
            "extra_historical_evidence_steps": [],
            "prefixes": prefix_rows,
            "constraint_graph_sha256": graph.canonical_sha256(),
            "constraint_graph": graph_rows,
            "constraint_reviews": {
                constraint.constraint_id: {
                    "dec_role": None,
                    "factor_by_step": _blank_factors(steps),
                    "note": "",
                }
                for constraint in graph.constraints
            },
            "missing_dec_constraints": [],
            "dec_mapping": [],
            "review_complete": False,
        }
        validate_review_row(row, require_complete=False, expected_mode=mode)
        rows.append(row)
    return rows


def _review_html(records: Sequence[Mapping[str, object]], *, mode: str) -> str:
    payload = json.dumps(list(records), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = "MF3ZP DEC 首轮校准" if mode == "first" else "MF3ZP DEC 盲重复审核"
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}} body{{margin:0;font:14px system-ui,sans-serif;background:#e9edf2;color:#17202a}}
#top{{position:sticky;top:0;z-index:5;background:#17202a;color:white;padding:8px 14px;display:flex;gap:12px;align-items:center}}
#top button,#top input{{padding:6px 10px}} #progress{{margin-left:auto}} main{{padding:12px}}
.card{{background:white;border-radius:10px;padding:12px;box-shadow:0 1px 5px #bcc4ce}}
.layout{{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(420px,1fr);gap:12px}}
.visuals{{min-width:0}} .frame-tabs{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}}
.frame-tabs button.active{{background:#2457a7;color:white}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.images img{{width:100%;height:66vh;object-fit:contain;background:#111;border-radius:6px}}
.side{{max-height:82vh;overflow:auto;padding-right:4px}} .instruction{{font-size:16px;line-height:1.45;background:#f6f8fa;padding:10px;border-radius:6px}}
.constraint{{border:1px solid #ccd4dd;border-radius:7px;padding:8px;margin:8px 0}} .meta{{color:#53606e;font-size:12px}}
.factor-grid{{display:grid;grid-template-columns:52px repeat(3,1fr);gap:4px;align-items:center;margin-top:6px}}
select,input,textarea,button{{font:inherit}} select,textarea{{width:100%}} textarea{{min-height:44px}}
.hidden{{display:none}} .warn{{color:#a12622}} .ok{{color:#176b34}} @media(max-width:1000px){{.layout{{grid-template-columns:1fr}}.images img{{height:45vh}}.side{{max-height:none}}}}
</style></head><body>
<div id="top"><b>{title}</b><button id="prev">上一条</button><button id="next">下一条</button>
<label>审核者 ID <input id="reviewer" autocomplete="off"></label><button id="export">导出 JSONL</button><span id="progress"></span></div>
<main><section class="card"><div class="layout"><div class="visuals"><div id="frameTabs" class="frame-tabs"></div><div class="images"><img id="story"><img id="pano"></div></div>
<aside class="side"><div id="identity"></div><h3>指令</h3><div id="instruction" class="instruction"></div><h3>当前候选（仅不透明 ID）</h3><div id="candidates"></div>
<p><b>顺序：</b>先判定每个约束与当前决策的关系；仅对 DEC_REQUIRED / PREREQUISITE_ONLY 填写各 prefix 的 S/G/E。U/A/D 由 K=3 自动派生。</p>
<div id="constraints"></div><button id="addMissing">添加缺失 DEC 原子</button><div id="missing"></div>
<label><input type="checkbox" id="complete"> 本事件审核完成</label><p id="message"></p></aside></div></section></main>
<script id="data" type="application/json">{payload}</script><script>
const KEY='mf3zp-dec-{mode}-v1'; const data=JSON.parse(document.getElementById('data').textContent);
let rows=JSON.parse(localStorage.getItem(KEY)||'null')||structuredClone(data), index=0, frameIndex=0;
const roles=['','DEC_REQUIRED','PREREQUISITE_ONLY','FUTURE_NOT_RELEVANT','REDUNDANT','INCORRECT'];
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function save(){{localStorage.setItem(KEY,JSON.stringify(rows));renderProgress()}}
function isDEC(v){{return v==='DEC_REQUIRED'||v==='PREREQUISITE_ONLY'}}
function renderProgress(){{document.getElementById('progress').textContent=`${{index+1}}/${{rows.length}} · 完成 ${{rows.filter(r=>r.review_complete).length}}`}}
function setFactor(cid,step,key,value){{let x=rows[index].constraint_reviews[cid].factor_by_step.find(v=>v.step===step);x[key]=value===''?null:value==='true';save()}}
function factorGrid(cid,item){{if(!(isDEC(item.dec_role)||item.role==='MISSING_DEC_CONSTRAINT'))return '';let h='<div class="factor-grid"><b>step</b><b>S</b><b>G</b><b>E</b>';for(const f of item.factor_by_step){{h+=`<span>${{f.step}}</span>`;for(const k of ['instantiated','distinguishable','resolved']){{h+=`<select data-cid="${{cid}}" data-step="${{f.step}}" data-factor="${{k}}"><option value=""></option><option value="true" ${{f[k]===true?'selected':''}}>是</option><option value="false" ${{f[k]===false?'selected':''}}>否</option></select>`}}}}return h+'</div>'}}
function renderConstraints(){{const r=rows[index], root=document.getElementById('constraints');root.innerHTML=r.constraint_graph.map(g=>{{const x=r.constraint_reviews[g.constraint_id];return `<div class="constraint"><b>${{esc(g.constraint_id)}} · ${{esc(g.kind)}}</b><div>${{esc(g.subject)}} ${{esc(g.relation||'')}} ${{esc(g.object||'')}}</div><div class="meta">依赖: ${{esc((g.dependencies||[]).join(', ')||'-')}}</div><select class="role" data-cid="${{g.constraint_id}}">${{roles.map(v=>`<option value="${{v}}" ${{x.dec_role===v?'selected':''}}>${{v||'请选择关系'}}</option>`).join('')}}</select>${{factorGrid(g.constraint_id,x)}}<textarea class="note" data-cid="${{g.constraint_id}}" placeholder="可选备注">${{esc(x.note)}}</textarea></div>`}}).join('');
root.querySelectorAll('.role').forEach(el=>el.onchange=e=>{{const x=r.constraint_reviews[e.target.dataset.cid];x.dec_role=e.target.value||null;if(!isDEC(x.dec_role))x.factor_by_step.forEach(f=>{{f.instantiated=f.distinguishable=f.resolved=null}});save();renderConstraints()}});
root.querySelectorAll('[data-factor]').forEach(el=>el.onchange=e=>setFactor(e.target.dataset.cid,Number(e.target.dataset.step),e.target.dataset.factor,e.target.value));
root.querySelectorAll('.note').forEach(el=>el.oninput=e=>{{r.constraint_reviews[e.target.dataset.cid].note=e.target.value;save()}})}}
function renderMissing(){{const r=rows[index], ids=r.constraint_graph.map(x=>x.constraint_id);document.getElementById('missing').innerHTML=r.missing_dec_constraints.map((m,i)=>`<div class="constraint"><b>${{esc(m.human_dec_item_id)}}</b><textarea data-mi="${{i}}" class="mtext" placeholder="缺失的决定性证据原子">${{esc(m.text)}}</textarea><label>人工映射（不使用 embedding 自动匹配）<select data-map="${{i}}"><option value="">无对应 Qwen 原子</option>${{ids.map(cid=>`<option value="${{cid}}" ${{m.qwen_constraint_id===cid?'selected':''}}>${{cid}}</option>`).join('')}}</select></label><label>匹配类型<select data-mtype="${{i}}"><option value="MISSING" ${{m.match_type==='MISSING'?'selected':''}}>MISSING</option><option value="SPLIT" ${{m.match_type==='SPLIT'?'selected':''}}>SPLIT</option><option value="MERGE" ${{m.match_type==='MERGE'?'selected':''}}>MERGE</option></select></label><button data-rm="${{i}}">删除</button>${{factorGrid('missing:'+i,m)}}</div>`).join('');document.querySelectorAll('[data-rm]').forEach(b=>b.onclick=e=>{{r.missing_dec_constraints.splice(Number(e.target.dataset.rm),1);save();renderMissing()}});document.querySelectorAll('.mtext').forEach(t=>t.oninput=e=>{{r.missing_dec_constraints[Number(e.target.dataset.mi)].text=e.target.value;save()}});document.querySelectorAll('[data-map]').forEach(el=>el.onchange=e=>{{const m=r.missing_dec_constraints[Number(e.target.dataset.map)];m.qwen_constraint_id=e.target.value||null;m.match_type=m.qwen_constraint_id?(m.match_type==='MISSING'?'SPLIT':m.match_type):'MISSING';save();renderMissing()}});document.querySelectorAll('[data-mtype]').forEach(el=>el.onchange=e=>{{const m=r.missing_dec_constraints[Number(e.target.dataset.mtype)];m.match_type=e.target.value;if(m.match_type==='MISSING')m.qwen_constraint_id=null;save();renderMissing()}});document.querySelectorAll('[data-factor]').forEach(el=>{{if(el.dataset.cid.startsWith('missing:'))el.onchange=e=>{{const m=r.missing_dec_constraints[Number(e.target.dataset.cid.split(':')[1])];const f=m.factor_by_step.find(v=>v.step===Number(e.target.dataset.step));f[e.target.dataset.factor]=e.target.value===''?null:e.target.value==='true';save()}}}})}}
function renderFrame(){{const r=rows[index],f=r.prefixes[frameIndex];document.querySelectorAll('#frameTabs button').forEach((b,i)=>b.classList.toggle('active',i===frameIndex));document.getElementById('story').src='../../../'+f.causal_storyboard_path;document.getElementById('pano').src='../../../'+f.current_panorama_path}}
function render(){{const r=rows[index];frameIndex=r.prefixes.length-1;document.getElementById('reviewer').value=r.reviewer_id;document.getElementById('identity').innerHTML=`<b>${{esc(r.dataset)}} · scene ${{esc(r.scene_id)}} · ep ${{esc(r.episode_id)}} · decision step ${{r.decision_step}}</b>`;document.getElementById('instruction').textContent=r.instruction;document.getElementById('candidates').textContent=r.current_candidate_ids.join(' · ');document.getElementById('frameTabs').innerHTML=r.prefixes.map((f,i)=>`<button data-fi="${{i}}">prefix ${{f.step}}</button>`).join('');document.querySelectorAll('[data-fi]').forEach(b=>b.onclick=e=>{{frameIndex=Number(e.target.dataset.fi);renderFrame()}});document.getElementById('complete').checked=r.review_complete;renderConstraints();renderMissing();renderFrame();renderProgress()}}
function buildMappings(r){{const out=[];for(const [cid,x] of Object.entries(r.constraint_reviews))if(isDEC(x.dec_role))out.push({{human_dec_item_id:'human::'+cid,qwen_constraint_id:cid,match_type:'EXACT_QWEN_ATOM'}});for(const m of r.missing_dec_constraints)out.push({{human_dec_item_id:m.human_dec_item_id,qwen_constraint_id:m.qwen_constraint_id,match_type:m.match_type}});return out}}
document.getElementById('reviewer').oninput=e=>{{rows.forEach(r=>r.reviewer_id=e.target.value);save()}};document.getElementById('complete').onchange=e=>{{rows[index].review_complete=e.target.checked;save()}};
document.getElementById('prev').onclick=()=>{{index=(index+rows.length-1)%rows.length;render()}};document.getElementById('next').onclick=()=>{{index=(index+1)%rows.length;render()}};
document.getElementById('addMissing').onclick=()=>{{const r=rows[index],n=r.missing_dec_constraints.length+1;r.missing_dec_constraints.push({{human_dec_item_id:`missing::${{r.event_id.slice(0,8)}}::${{n}}`,role:'MISSING_DEC_CONSTRAINT',text:'',qwen_constraint_id:null,match_type:'MISSING',factor_by_step:r.prefixes.map(f=>({{step:f.step,instantiated:null,distinguishable:null,resolved:null}})),note:''}});save();renderMissing()}};
document.getElementById('export').onclick=()=>{{for(const r of rows)r.dec_mapping=buildMappings(r);const blob=new Blob([rows.map(r=>JSON.stringify(r)).join('\n')+'\n'],{{type:'application/jsonl'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='MF3ZP_SINGLE_EXPERT_{'RETEST' if mode == 'retest' else 'FIRST'}_COMPLETED.jsonl';a.click();URL.revokeObjectURL(a.href)}};render();
</script></body></html>'''


def prepare_first_review() -> dict[str, object]:
    verify_scout_protocol()
    rows = build_review_rows(mode="first")
    atomic_jsonl(REVIEW_TEMPLATE, rows)
    atomic_text(REVIEW_HTML, _review_html(rows, mode="first"))
    status = {
        "schema_version": "revealnav-mf3zp-single-expert-status/1",
        "revision": REVISION,
        "status": "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_AWAITING_HUMAN_REVIEW",
        "protocol_sha256": sha256_file(SCOUT_PROTOCOL),
        "review_events": len(rows),
        "review_template": inventory(REVIEW_TEMPLATE),
        "review_html": inventory(REVIEW_HTML),
        "retest_ids_sealed": True,
        "retest_package_materialized": False,
        "qwen_api_calls": 0,
        "human_labels_fabricated": False,
        "oracle_headroom_run": False,
        "ree_training_run": False,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    atomic_json(STATUS_PATH, status)
    return status


_FROZEN_REVIEW_KEYS = {
    "schema_version", "review_mode", "reviewer_blinded_to_outcomes",
    "reviewer_blinded_to_qwen_factors", "event_id", "dataset", "scene_id",
    "episode_id", "instruction", "decision_step", "current_candidate_ids",
    "review_prefix_start", "review_prefix_end", "extra_historical_evidence_steps",
    "prefixes", "constraint_graph_sha256", "constraint_graph",
}


def validate_frozen_review_fields(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for key in _FROZEN_REVIEW_KEYS:
        if actual[key] != expected[key]:
            raise ScoutError(f"completed review mutated frozen field: {key}")


def load_completed_review(
    path: Path,
    *,
    mode: str,
) -> list[dict[str, object]]:
    selection = read_json(SCOUT_SELECTION if mode == "first" else RETEST_SELECTION)
    expected_rows = build_review_rows(mode=mode, selection=selection)
    expected_ids = [str(row["event_id"]) for row in expected_rows]
    rows = read_jsonl(path)
    if [str(row.get("event_id")) for row in rows] != list(expected_ids):
        raise ScoutError(f"{mode} review event population/order drift")
    normalized = [validate_review_row(row, require_complete=True, expected_mode=mode) for row in rows]
    for actual, expected in zip(normalized, expected_rows, strict=True):
        validate_frozen_review_fields(actual, expected)
    reviewer_ids = {str(row["reviewer_id"]) for row in normalized}
    if len(reviewer_ids) != 1:
        raise ScoutError("one stable expert ID is required")
    return normalized


def validate_first_review(path: Path) -> dict[str, object]:
    verify_scout_protocol()
    rows = load_completed_review(path, mode="first")
    marker = {
        "schema_version": "revealnav-mf3zp-single-expert-first-validation/1",
        "status": "FIRST_REVIEW_STRUCTURALLY_VALID",
        "review_file": inventory(path),
        "event_count": len(rows),
        "reviewer_id_sha256": hashlib.sha256(str(rows[0]["reviewer_id"]).encode()).hexdigest(),
        "labels_used_for_retest_selection": False,
        "public_split_access": PUBLIC_CLOSED,
    }
    atomic_json(FIRST_VALIDATION, marker)
    return marker


def prepare_retest() -> dict[str, object]:
    verify_scout_protocol()
    marker = read_json(FIRST_VALIDATION)
    if marker.get("status") != "FIRST_REVIEW_STRUCTURALLY_VALID":
        raise ScoutError("first review structural validation is required")
    rows = build_review_rows(mode="retest")
    atomic_jsonl(RETEST_TEMPLATE, rows)
    atomic_text(RETEST_HTML, _review_html(rows, mode="retest"))
    return {
        "status": "MF3ZP_SINGLE_EXPERT_RETEST_AWAITING_HUMAN_REVIEW",
        "event_count": len(rows),
        "template": inventory(RETEST_TEMPLATE),
        "html": inventory(RETEST_HTML),
        "first_pass_label_file_read": False,
        "qwen_api_calls": 0,
        "public_split_access": PUBLIC_CLOSED,
    }


def _combined_evidence(request_id: str) -> dict[str, object]:
    v11 = EVIDENCE_V11_DIR / f"{request_id}.json"
    base = EVIDENCE_DIR / f"{request_id}.json"
    path = v11 if v11.is_file() else base
    record = read_json(path)
    if record.get("source_request_id") != request_id or record.get("human_verified") is not False or record.get("gold") is not False:
        raise ScoutError("Qwen evidence provenance drift")
    return record


def _cohen_kappa(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> float:
    if not pairs:
        raise ScoutError("kappa has no aligned items")
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[c] / n) * (right[c] / n) for c in categories)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _confusion(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> dict[str, dict[str, int]]:
    result = {truth: {pred: 0 for pred in categories} for truth in categories}
    for truth, pred in pairs:
        result[truth][pred] += 1
    return result


def _accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")


def _balanced_accuracy(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> float:
    recalls = []
    for category in categories:
        subset = [pred for truth, pred in pairs if truth == category]
        if subset:
            recalls.append(sum(pred == category for pred in subset) / len(subset))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def dec_adequacy_counts(
    *,
    qwen_proposed_ids: Iterable[str],
    human_dec_item_count: int,
    mapped_qwen_ids: Iterable[str],
) -> dict[str, float | int]:
    qwen = set(qwen_proposed_ids)
    mapped = set(mapped_qwen_ids)
    if not mapped <= qwen or human_dec_item_count < len(mapped):
        raise ScoutError("invalid manual DEC mapping")
    matched = len(mapped)
    return {
        "qwen_proposed_count": len(qwen),
        "human_dec_count": human_dec_item_count,
        "matched_count": matched,
        "precision": matched / len(qwen) if qwen else float("nan"),
        "recall": matched / human_dec_item_count if human_dec_item_count else float("nan"),
    }


def false_decisive_summary(
    expert_qwen_uad_pairs: Sequence[tuple[str, str]],
    *,
    minimum_support: int = MIN_QWEN_D_SUPPORT,
) -> dict[str, object]:
    qwen_d = sum(prediction == "D" for _, prediction in expert_qwen_uad_pairs)
    false_d = sum(
        prediction == "D" and truth != "D"
        for truth, prediction in expert_qwen_uad_pairs
    )
    rate = false_d / qwen_d if qwen_d else float("nan")
    support = (
        "SUFFICIENT_QWEN_D_SUPPORT"
        if qwen_d >= minimum_support else "INSUFFICIENT_QWEN_D_SUPPORT"
    )
    return {
        "support_status": support,
        "minimum_support": minimum_support,
        "qwen_D_count": qwen_d,
        "false_D_count": false_d,
        "rate": rate,
    }


def _factor_tuple(item: Mapping[str, object]) -> tuple[list[bool], list[bool], list[bool]]:
    rows = item["factor_by_step"]
    return tuple([bool(row[key]) for row in rows] for key in ("instantiated", "distinguishable", "resolved"))  # type: ignore[return-value]


def _human_dec_items(row: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result = {
        f"human::{cid}": item
        for cid, item in row["constraint_reviews"].items()
        if DecRole(item["dec_role"]) in DEC_ROLES
    }
    for item in row["missing_dec_constraints"]:
        item_id = "human::missing::" + stable_sha256({
            "text": " ".join(str(item["text"]).casefold().split()),
            "qwen_constraint_id": item["qwen_constraint_id"],
            "match_type": item["match_type"],
        })
        if item_id in result:
            raise ScoutError("duplicate exact missing DEC atom")
        result[item_id] = item
    return result


def score_completed_reviews(
    first_rows: Sequence[Mapping[str, object]],
    retest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    first = {str(row["event_id"]): row for row in first_rows}
    retest = {str(row["event_id"]): row for row in retest_rows}
    if set(retest) - set(first):
        raise ScoutError("retest contains an event outside the first review")
    uad_retest_pairs: list[tuple[str, str]] = []
    e_retest_pairs: list[tuple[str, str]] = []
    for event_id, later in retest.items():
        earlier = first[event_id]
        earlier_items = _human_dec_items(earlier)
        later_items = _human_dec_items(later)
        for item_id in sorted(set(earlier_items) | set(later_items)):
            a = earlier_items.get(item_id)
            b = later_items.get(item_id)
            a_uad = (
                derive_constraint_uad(*_factor_tuple(a), stability_k=3)[-1].value
                if a is not None else "NOT_DEC"
            )
            b_uad = (
                derive_constraint_uad(*_factor_tuple(b), stability_k=3)[-1].value
                if b is not None else "NOT_DEC"
            )
            a_e = str(a["factor_by_step"][-1]["resolved"]) if a is not None else "NOT_DEC"
            b_e = str(b["factor_by_step"][-1]["resolved"]) if b is not None else "NOT_DEC"
            uad_retest_pairs.append((a_uad, b_uad))
            e_retest_pairs.append((a_e, b_e))
    intra = {
        "aligned_dec_constraints": len(uad_retest_pairs),
        "uad_kappa": _cohen_kappa(uad_retest_pairs, ("NOT_DEC", "U", "A", "D")),
        "e_kappa": _cohen_kappa(e_retest_pairs, ("NOT_DEC", "False", "True")),
        "dec_membership_disagreements": sum(
            "NOT_DEC" in pair and pair[0] != pair[1]
            for pair in uad_retest_pairs
        ),
    }

    tasks = _task_index()
    events = _event_by_id()
    qwen_total = human_total = intersection = 0
    s_pairs: list[tuple[str, str]] = []
    g_pairs: list[tuple[str, str]] = []
    e_pairs: list[tuple[str, str]] = []
    uad_pairs: list[tuple[str, str]] = []
    per_event_diag = []
    for row in first_rows:
        event = events[str(row["event_id"])]
        graph_ids = [str(item["constraint_id"]) for item in row["constraint_graph"]]
        dec_qwen_ids = {
            cid for cid, item in row["constraint_reviews"].items()
            if DecRole(item["dec_role"]) in DEC_ROLES
        }
        mappings = {
            str(item["qwen_constraint_id"])
            for item in row["dec_mapping"] if item["qwen_constraint_id"] is not None
        }
        missing = list(row["missing_dec_constraints"])
        qwen_total += len(graph_ids)
        human_total += len(dec_qwen_ids) + len(missing)
        intersection += len(mappings)
        future = sum(item["dec_role"] == "FUTURE_NOT_RELEVANT" for item in row["constraint_reviews"].values())
        redundant = sum(item["dec_role"] == "REDUNDANT" for item in row["constraint_reviews"].values())
        per_event_diag.append({
            "event_id": row["event_id"],
            "whole_instruction_constraint_count": len(graph_ids),
            "qwen_DEC_constraint_count": len(graph_ids),
            "expert_DEC_constraint_count": len(dec_qwen_ids) + len(missing),
            "future_irrelevant_fraction": future / len(graph_ids),
            "redundant_fraction": redundant / len(graph_ids),
            "compression": 1.0 - (len(dec_qwen_ids) + len(missing)) / len(graph_ids),
        })
        qwen_by_cid: dict[str, tuple[list[bool], list[bool], list[bool]]] = {
            cid: ([], [], []) for cid in dec_qwen_ids
        }
        for prefix in row["prefixes"]:
            key = (
                str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]),
                str(event["source_observation_stream_id"]), int(prefix["step"]),
            )
            request_id = str(tasks[key]["request_id"])
            qwen = _combined_evidence(request_id)["normalized_constraints"]
            for cid in dec_qwen_ids:
                human = row["constraint_reviews"][cid]["factor_by_step"]
                human_at_step = next(item for item in human if item["step"] == prefix["step"])
                q = qwen[cid]
                for pairs, key_name in ((s_pairs, "instantiated"), (g_pairs, "distinguishable"), (e_pairs, "resolved")):
                    pairs.append((str(human_at_step[key_name]), str(q[key_name])))
                for target, key_name in zip(qwen_by_cid[cid], ("instantiated", "distinguishable", "resolved"), strict=True):
                    target.append(bool(q[key_name]))
        for cid in dec_qwen_ids:
            human_state = derive_constraint_uad(*_factor_tuple(row["constraint_reviews"][cid]), stability_k=3)[-1].value
            qwen_state = derive_constraint_uad(*qwen_by_cid[cid], stability_k=3)[-1].value
            uad_pairs.append((human_state, qwen_state))

    dec = dec_adequacy_counts(
        qwen_proposed_ids=(f"qwen-{index}" for index in range(qwen_total)),
        human_dec_item_count=human_total,
        mapped_qwen_ids=(f"qwen-{index}" for index in range(intersection)),
    )
    agreement = {}
    for name, pairs, cats in (
        ("S", s_pairs, ("False", "True")),
        ("G", g_pairs, ("False", "True")),
        ("E", e_pairs, ("False", "True")),
        ("UAD", uad_pairs, ("U", "A", "D")),
    ):
        agreement[name] = {
            "count": len(pairs),
            "accuracy": _accuracy(pairs),
            "balanced_accuracy": _balanced_accuracy(pairs, cats),
            "confusion_expert_truth_qwen_prediction": _confusion(pairs, cats),
        }
    false_d_summary = false_decisive_summary(uad_pairs)
    thresholds = {
        "intra_uad_kappa": intra["uad_kappa"] >= 0.75,
        "intra_e_kappa": intra["e_kappa"] >= 0.80,
        "dec_precision": dec["precision"] >= 0.80,
        "dec_recall": dec["recall"] >= 0.90,
        "qwen_expert_uad_accuracy": agreement["UAD"]["accuracy"] >= 0.80,
        "false_decisive_rate": (
            false_d_summary["support_status"] == "SUFFICIENT_QWEN_D_SUPPORT"
            and false_d_summary["rate"] <= 0.10
        ),
    }
    ready = all(thresholds.values())
    return {
        "status": "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_READY_FOR_MULTI_REVIEW" if ready else "MF3ZP_SINGLE_EXPERT_DEC_SCOUT_NOT_READY",
        "formal_label_validity_pass": False,
        "intra_expert": intra,
        "dec_adequacy": dec,
        "qwen_expert_agreement": agreement,
        "false_decisive": false_d_summary,
        "readiness_checks": thresholds,
        "whole_instruction_vs_dec_diagnostic": per_event_diag,
        "oracle_headroom_authorized": False,
        "ree_training_authorized": False,
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }


__all__ = [
    "BASE_REVIEW_COMMIT", "CLOSURE_PROTOCOL", "EXPECTED_HISTORICAL_SHA256",
    "FIRST_VALIDATION", "IMPLEMENTATION_FILES", "OUTPUT", "PUBLIC_CLOSED",
    "RESULT_PATH", "RETEST_HTML", "RETEST_SELECTION", "RETEST_TEMPLATE",
    "REVIEW_HTML", "REVIEW_TEMPLATE", "REVISION", "SCOUT_PROTOCOL",
    "SCOUT_SELECTION", "STATUS_PATH", "ScoutError", "atomic_json",
    "atomic_jsonl", "build_closure_protocol", "build_review_rows",
    "build_selection_artifacts", "build_scout_protocol", "inventory",
    "dec_adequacy_counts", "false_decisive_summary",
    "prepare_first_review", "prepare_retest", "read_json", "read_jsonl",
    "load_completed_review",
    "score_completed_reviews", "seal_protocols", "select_retest_events",
    "select_scout_events", "sha256_file", "stable_sha256",
    "validate_first_review", "validate_frozen_review_fields", "verify_closure", "verify_scout_protocol",
    "write_selection_artifacts",
]
