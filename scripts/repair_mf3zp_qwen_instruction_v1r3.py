#!/usr/bin/env python3
"""Outcome-blind structural repair for two invalid instruction DAG responses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revealnav_mf3.qwen_evidence_annotation import INSTRUCTION_SYSTEM_PROMPT
V1R2_PATH = ROOT / "scripts/annotate_mf3zp_qwen_transport_v1r2.py"
V1R2_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_TRANSPORT_V1R2_PROTOCOL.json"
PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_INSTRUCTION_REPAIR_V1R3_PROTOCOL.json"
ADDENDUM = """STRUCTURAL REPAIR REQUIREMENT: Keep the semantic constraint DAG
definition unchanged. Before returning JSON, construct the top-level
dependencies list mechanically from every constraint's dependencies field: for
each dependency d of constraint c output exactly one {\"from\":d,\"to\":c};
output no other edges and no duplicates."""


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


v2 = load(V1R2_PATH, "mf3zp_transport_v1r2_parent")
base = v2.base


def missing_instructions():
    events = base.read_events()
    values = sorted({str(event["instruction"]).strip() for event in events}, key=base._instruction_key)
    return [value for value in values if not (base.INSTRUCTION_DIR / f"{base._instruction_key(value)}.json").is_file()]


def protocol_value(selected=None):
    v2.verify(); missing = missing_instructions() if selected is None else list(selected)
    if len(missing) != 2: raise RuntimeError(f"repair protocol requires exactly two selected DAGs, observed {len(missing)}")
    return {
        "schema_version": "revealnav-mf3zp-qwen-instruction-repair/1",
        "revision": "mf3zp_revealskill_qwen_instruction_v1r3",
        "status": "SEALED_BEFORE_QWEN_INSTRUCTION_REPAIR_RESULTS",
        "science_protocol": v2.v1.inventory(v2.v1.SCIENCE_PROTOCOL),
        "parent_transport_protocol": v2.v1.inventory(V1R2_PROTOCOL),
        "repair_entrypoint": v2.v1.inventory(Path(__file__).resolve()),
        "model": "qwen3.8-max",
        "base_prompt_sha256": hashlib.sha256(INSTRUCTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "repair_addendum": ADDENDUM,
        "repair_addendum_sha256": hashlib.sha256(ADDENDUM.encode("utf-8")).hexdigest(),
        "missing_instruction_sha256": [base._instruction_key(value) for value in missing],
        "selection_reason": "strict top-level dependency-edge consistency failure only",
        "semantic_schema_change": False,
        "population_change": False,
        "outcome_access": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }


def seal():
    value = protocol_value()
    if PROTOCOL.exists() or PROTOCOL.is_symlink(): raise RuntimeError("refusing to overwrite repair protocol")
    partial = PROTOCOL.with_name(PROTOCOL.name + ".part"); partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(partial, PROTOCOL); return value


def verify():
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    instructions = {base._instruction_key(item): item for item in sorted({str(event["instruction"]).strip() for event in base.read_events()})}
    try: selected = [instructions[key] for key in value["missing_instruction_sha256"]]
    except KeyError as error: raise RuntimeError("repair selection left sealed pilot") from error
    if value != protocol_value(selected): raise RuntimeError("repair protocol/source/selection drift")
    return value


def run_one(api_key: str, instruction: str):
    payload = base.instruction_request(instruction)
    payload = dict(payload); messages = [dict(item) for item in payload["messages"]]; messages[0]["content"] = str(messages[0]["content"]) + "\n\n" + ADDENDUM; payload["messages"] = messages
    response, provider_model, provider_id = v2.api_request(api_key, payload)
    graph = base.parse_instruction_response(response, instruction=instruction)
    record = base.request_record(stage="instruction", payload=payload, response=response, provider_model=provider_model)
    effective = dict(payload); effective["enable_thinking"] = False; effective["max_tokens"] = v2.MAX_TOKENS
    record.update({
        "schema_version": "revealnav-mf3zp-qwen-instruction-preannotation/1", "status": "PROVISIONAL_QWEN_PREANNOTATION",
        "instruction_sha256": base._instruction_key(instruction), "instruction": instruction, "constraint_graph_sha256": graph.canonical_sha256(),
        "provider_request_id": provider_id, "effective_request_payload_sha256": base.stable_sha256(effective),
        "transport_protocol_sha256": v2.v1.sha256_file(V1R2_PROTOCOL), "repair_protocol_sha256": v2.v1.sha256_file(PROTOCOL),
        "enable_thinking": False, "max_tokens": v2.MAX_TOKENS, "human_verified": False, "gold": False,
    })
    base.atomic_json(base.INSTRUCTION_DIR / f"{base._instruction_key(instruction)}.json", record)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("seal", "verify", "run")); args = parser.parse_args()
    if args.command == "seal": value=seal(); print(json.dumps({"status":value["status"],"sha256":v2.v1.sha256_file(PROTOCOL)},indent=2)); return 0
    verify()
    if args.command == "verify": print(json.dumps({"status":"MF3ZP_QWEN_INSTRUCTION_REPAIR_V1R3_VERIFIED"},indent=2)); return 0
    key=base.read_api_key(); failures={}
    sealed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    all_instructions = {base._instruction_key(item): item for item in {str(event["instruction"]).strip() for event in base.read_events()}}
    for instruction in (all_instructions[key] for key in sealed["missing_instruction_sha256"]):
        if (base.INSTRUCTION_DIR / f"{base._instruction_key(instruction)}.json").is_file():
            continue
        try: run_one(key,instruction)
        except Exception as error: failures[base._instruction_key(instruction)]=f"{type(error).__name__}: {error}"
    result=base.status(); print(json.dumps({"failures":failures,"status":result},indent=2,ensure_ascii=False)); return 0 if not failures else 2


if __name__ == "__main__": raise SystemExit(main())
