import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from revealnav_mf3.mf3zu_evidence_memory import (
    MF3ZUContractError,
    QWEN_ENABLE_THINKING,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    parse_instruction_response,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recovery = _load_script(
    "mf3zu_annotation_recovery_v1r1_test_module",
    "scripts/run_mf3zu_annotation_recovery_v1r1.py",
)


def _instruction_response(count: int) -> dict[str, object]:
    return {
        "instruction_atoms": [
            {
                "instruction_atom_id": f"a{index:02d}",
                "text": f"constraint {index}",
                "semantic_kind": "DIRECTION",
                "depends_on": [] if index == 1 else [f"a{index - 1:02d}"],
            }
            for index in range(1, count + 1)
        ]
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class Mf3zuAnnotationRecoveryV1r1Test(unittest.TestCase):
    def test_parent_32_rejects_33_and_v1r1_accepts_through_99(self):
        instruction = "Follow every instruction constraint in order."
        with self.assertRaises(MF3ZUContractError):
            parse_instruction_response(
                _instruction_response(33), instruction=instruction
            )
        graph_33 = recovery.parse_instruction_response_v1r1(
            _instruction_response(33),
            instruction=instruction,
        )
        graph_99 = recovery.parse_instruction_response_v1r1(
            _instruction_response(99),
            instruction=instruction,
        )
        self.assertEqual(len(graph_33.atoms), 33)
        self.assertEqual(len(graph_99.atoms), 99)
        self.assertIsInstance(graph_33, recovery.RecoveryInstructionGraph)
        self.assertEqual(recovery.ATOM_LIMIT, 99)
        with self.assertRaises(MF3ZUContractError):
            recovery.parse_instruction_response_v1r1(
                _instruction_response(100),
                instruction=instruction,
            )

    def test_parent_sealed_protocol_and_inventory_still_verify(self):
        # The recovery test filename deliberately does not match the parent's
        # sealed tests/test_mf3zu_*.py implementation-inventory glob.
        from revealnav_mf3 import mf3zu_protocol

        value = mf3zu_protocol.verify_protocol()
        self.assertEqual(value["revision"], mf3zu_protocol.REVISION)
        self.assertNotIn(
            "tests/test_annotation_recovery_mf3zu_v1r1.py",
            value["implementation_inventory"],
        )
        for item in value["implementation_inventory"].values():
            self.assertEqual(
                mf3zu_protocol.inventory(ROOT / str(item["path"])),
                item,
            )

    def _parent_failure_fixture(self, base: Path) -> tuple[Path, list[dict], set[str]]:
        parent_root = base / "parent"
        requests = [
            {
                "request_id": f"request-{index:03d}",
                "episode_id": f"episode-{index:03d}",
                "instruction": f"instruction {index}",
                "payload": {},
            }
            for index in range(154)
        ]
        failed_ids = {
            str(row["request_id"])
            for row in requests[-12:]
        }
        request_path = parent_root / "MF3ZU_INSTRUCTION_REQUESTS.jsonl"
        _write_jsonl(request_path, requests)

        with mock.patch.object(recovery, "ROOT", base):
            request_inventory = recovery.inventory(request_path)
        _write_json(parent_root / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json", {
            "revision": recovery.PARENT_REVISION,
            "status": "SEALED_BEFORE_INSTRUCTION_RESPONSES",
            "episodes": 154,
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": QWEN_MAX_TOKENS,
            "thinking": QWEN_ENABLE_THINKING,
            "requests": request_inventory,
            "ranking_label_read": False,
            "task_metric_read": False,
            "public_split_access": False,
        })
        failures = []
        for row in requests:
            request_id = str(row["request_id"])
            value = {
                "revision": recovery.PARENT_REVISION,
                "request_id": request_id,
                "status": "PASS",
                "response": _instruction_response(1),
                "model_requested": QWEN_MODEL,
                "provider_model": QWEN_MODEL,
                "temperature": QWEN_TEMPERATURE,
                "max_tokens": QWEN_MAX_TOKENS,
                "thinking": QWEN_ENABLE_THINKING,
                "ranking_label_read": False,
                "task_metric_read": False,
                "public_split_access": False,
            }
            if request_id in failed_ids:
                value.pop("response")
                value["status"] = "FAIL"
                value["error"] = (
                    "MF3ZUContractError: invalid instruction atom list"
                )
                failures.append({"request_id": request_id})
            _write_json(
                parent_root / "responses/instruction" / f"{request_id}.json",
                value,
            )
        _write_json(
            parent_root / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json",
            {
                "revision": recovery.PARENT_REVISION,
                "status": "FAIL",
                "planned": 154,
                "response_files": 154,
                "pass": 142,
                "model": QWEN_MODEL,
                "failures": failures,
                "ranking_label_read": False,
                "task_metric_read": False,
                "public_split_access": False,
            },
        )
        # A deliberately malformed decoy proves that repair selection does not
        # need, inventory, or parse the separate target-bearing artifact.
        (parent_root / "MF3ZU_RXR_EXACT_TARGETS.jsonl").write_bytes(
            b"not-json-and-must-never-be-opened\xff"
        )
        return parent_root, requests, failed_ids

    def test_repair_selection_uses_only_parent_manifest_and_responses(self):
        source = (
            ROOT / "scripts/run_mf3zu_annotation_recovery_v1r1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("MF3ZU_RXR_EXACT_TARGETS", source)
        self.assertNotIn("EXACT_TARGETS_PATH", source)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            parent_root, requests, failed_ids = self._parent_failure_fixture(base)
            original_path_open = Path.open

            def guarded_open(path: Path, *args, **kwargs):
                if path.name == "MF3ZU_RXR_EXACT_TARGETS.jsonl":
                    raise AssertionError("exact target artifact was opened")
                return original_path_open(path, *args, **kwargs)

            with (
                mock.patch.object(recovery, "ROOT", base),
                mock.patch.object(recovery, "PARENT_OUTPUT", parent_root),
                mock.patch.object(Path, "open", guarded_open),
            ):
                state = recovery.classify_parent()

            self.assertEqual(len(state["requests"]), 154)
            self.assertEqual(len(state["passed"]), 142)
            self.assertEqual(len(state["failed"]), 12)
            self.assertEqual(
                {str(row["request_id"]) for row in state["failed"]},
                failed_ids,
            )
            self.assertEqual(
                {str(row["request_id"]) for row in state["requests"]},
                {str(row["request_id"]) for row in requests},
            )

    def test_merged_view_is_complete_and_preserves_parent_pass_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            parent_root = base / "parent"
            output = base / "recovery"
            output.mkdir(parents=True)
            requests = [
                {
                    "request_id": f"request-{index:03d}",
                    "episode_id": f"episode-{index:03d}",
                    "instruction": f"instruction {index}",
                    "payload": {},
                }
                for index in range(154)
            ]
            failed_ids = {
                str(row["request_id"])
                for row in requests[-12:]
            }
            _write_json(output / recovery.PROTOCOL_NAME, {"status": "sealed"})
            protocol_sha256 = hashlib.sha256(
                (output / recovery.PROTOCOL_NAME).read_bytes()
            ).hexdigest()
            parent_snapshots: dict[str, bytes] = {}
            for row in requests:
                request_id = str(row["request_id"])
                if request_id in failed_ids:
                    atom_count = 33 if request_id == sorted(failed_ids)[0] else 1
                    path = (
                        output / "responses/instruction_repair"
                        / f"{request_id}.json"
                    )
                else:
                    atom_count = 1
                    path = (
                        parent_root / "responses/instruction"
                        / f"{request_id}.json"
                    )
                response = {
                    "status": "PASS",
                    "request_id": request_id,
                    "response": _instruction_response(atom_count),
                }
                if request_id in failed_ids:
                    response.update({
                        "revision": recovery.REVISION,
                        "episode_id": str(row["episode_id"]),
                        "model_requested": QWEN_MODEL,
                        "provider_model": QWEN_MODEL,
                        "instruction_atom_limit": 99,
                        "request_payload_sha256": recovery.stable_sha256(
                            row["payload"]
                        ),
                        "repair_protocol_sha256": protocol_sha256,
                        "ranking_label_read": False,
                        "task_metric_read": False,
                        "public_split_access": False,
                    })
                    _write_json(
                        output / "intents/instruction_repair"
                        / f"{request_id}.json",
                        recovery._request_intent_value(
                            stage="instruction_repair",
                            request_id=request_id,
                            identity=str(row["episode_id"]),
                            request_payload_sha256=recovery.stable_sha256(
                                row["payload"]
                            ),
                            protocol_sha256=protocol_sha256,
                        ),
                    )
                _write_json(path, response)
                if request_id not in failed_ids:
                    parent_snapshots[request_id] = path.read_bytes()

            for name in (
                "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json",
                "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json",
                "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json",
                "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json",
            ):
                _write_json(parent_root / name, {"name": name})
            _write_json(output / recovery.REPAIR_MANIFEST_NAME, {
                "revision": recovery.REVISION,
                "status": "PASS",
                "planned": 12,
                "pass": 12,
                "instruction_atom_limit": 99,
            })
            state = {
                "requests": requests,
                "failed": [
                    {"request_id": request_id}
                    for request_id in sorted(failed_ids)
                ],
            }

            with (
                mock.patch.object(recovery, "ROOT", base),
                mock.patch.object(recovery, "PARENT_OUTPUT", parent_root),
                mock.patch.object(recovery, "verify_protocol", return_value={}),
                mock.patch.object(recovery, "classify_parent", return_value=state),
            ):
                manifest = recovery.materialize_merged_view(output)
                index = recovery.jsonl(output / recovery.MERGED_INDEX_NAME)

            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["planned"], 154)
            self.assertEqual(manifest["pass"], 154)
            self.assertEqual(manifest["instruction_atom_limit"], 99)
            self.assertEqual(len(index), 154)
            self.assertEqual(len({row["request_id"] for row in index}), 154)
            self.assertEqual(len({row["episode_id"] for row in index}), 154)
            self.assertEqual(
                sum(row["source_kind"] == "parent_pass" for row in index),
                142,
            )
            self.assertEqual(
                sum(row["source_kind"] == "v1r1_repair" for row in index),
                12,
            )
            self.assertIn(33, {row["instruction_atom_count"] for row in index})
            for request_id, before in parent_snapshots.items():
                source = (
                    parent_root / "responses/instruction" / f"{request_id}.json"
                )
                merged = output / "responses/instruction" / f"{request_id}.json"
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(merged.read_bytes(), before)
                self.assertEqual(
                    hashlib.sha256(merged.read_bytes()).hexdigest(),
                    hashlib.sha256(before).hexdigest(),
                )

    def test_invalid_evidence_response_is_retained_and_not_retried(self):
        graph = parse_instruction_response(
            _instruction_response(1), instruction="turn left"
        )
        row = {
            "request_id": "evidence-request",
            "event_id": "event",
            "decision_step": 1,
            "candidate_alias_to_action_id": {"C00": "g0", "C01": "g1"},
        }
        payload = {
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": QWEN_MAX_TOKENS,
            "enable_thinking": QWEN_ENABLE_THINKING,
            "messages": [],
        }
        raw = '{"atoms":[]}'
        provider = {
            "status": "PARSED_JSON",
            "provider_model": QWEN_MODEL,
            "response": {"atoms": []},
            "raw_content": raw,
            "usage": {"total_tokens": 1},
            "transport_attempts": [{"attempt": 1, "status": "PARSED_JSON"}],
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            with (
                mock.patch.object(
                    recovery.parent, "_evidence_payload", return_value=payload
                ),
                mock.patch.object(
                    recovery,
                    "_provider_request_preserving",
                    return_value=provider,
                ) as provider_call,
            ):
                first = recovery._annotate_evidence_one(
                    api_key="unused",
                    row=row,
                    graph=graph,
                    output=output,
                    protocol_sha256="a" * 64,
                )
                destination = (
                    output / "responses/evidence/evidence-request.json"
                )
                first_bytes = destination.read_bytes()
                first_sha = hashlib.sha256(first_bytes).hexdigest()
                second = recovery._annotate_evidence_one(
                    api_key="unused",
                    row=row,
                    graph=graph,
                    output=output,
                    protocol_sha256="a" * 64,
                )

            self.assertEqual(provider_call.call_count, 1)
            self.assertEqual(first["status"], "FAIL")
            self.assertEqual(first["raw_provider_content"], raw)
            self.assertEqual(first["invalid_parsed_response"], {"atoms": []})
            self.assertIn("MF3ZUContractError", str(first["error"]))
            self.assertEqual(second, first)
            self.assertEqual(destination.read_bytes(), first_bytes)
            self.assertEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(), first_sha
            )

    def test_ambiguous_intent_refuses_duplicate_provider_call(self):
        graph = parse_instruction_response(
            _instruction_response(1), instruction="turn left"
        )
        row = {
            "request_id": "ambiguous-request",
            "event_id": "event",
            "decision_step": 1,
            "candidate_alias_to_action_id": {"C00": "g0", "C01": "g1"},
        }
        payload = {
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": QWEN_MAX_TOKENS,
            "enable_thinking": QWEN_ENABLE_THINKING,
            "messages": [],
        }
        protocol_sha256 = "b" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            _write_json(
                output / "intents/evidence/ambiguous-request.json",
                recovery._request_intent_value(
                    stage="evidence",
                    request_id="ambiguous-request",
                    identity="event",
                    request_payload_sha256=recovery.stable_sha256(payload),
                    protocol_sha256=protocol_sha256,
                ),
            )
            with (
                mock.patch.object(
                    recovery.parent, "_evidence_payload", return_value=payload
                ),
                mock.patch.object(
                    recovery, "_provider_request_preserving"
                ) as provider_call,
                self.assertRaisesRegex(
                    recovery.RecoveryError, "refusing duplicate request"
                ),
            ):
                recovery._annotate_evidence_one(
                    api_key="unused",
                    row=row,
                    graph=graph,
                    output=output,
                    protocol_sha256=protocol_sha256,
                )
            provider_call.assert_not_called()

    def test_recoverable_deterministic_commit_and_fixed_cli_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            destination = base / "artifact.json"
            partial = destination.with_name(destination.name + ".part")
            value = {"status": "SEALED"}
            partial.write_bytes(recovery._json_bytes(value))
            recovery.recoverable_atomic_json(destination, value)
            self.assertFalse(partial.exists())
            before = destination.read_bytes()
            recovery.recoverable_atomic_json(destination, value)
            self.assertEqual(destination.read_bytes(), before)
            with self.assertRaisesRegex(
                recovery.RecoveryError, "deterministic artifact differs"
            ):
                recovery.recoverable_atomic_json(
                    destination, {"status": "DIFFERENT"}
                )
            with self.assertRaisesRegex(
                recovery.RecoveryError, "output root is fixed"
            ):
                recovery._validate_cli_output_root(base.resolve())
            with self.assertRaisesRegex(
                recovery.RecoveryError, "output root is fixed"
            ):
                recovery._validate_cli_output_root(
                    recovery.PARENT_OUTPUT.resolve()
                )


if __name__ == "__main__":
    unittest.main()
