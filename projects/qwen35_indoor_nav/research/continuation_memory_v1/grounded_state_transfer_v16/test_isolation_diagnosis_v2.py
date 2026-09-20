"""CPU acceptance tests for isolation_diagnosis_v2.

The fixtures live in a temporary directory and are never placed below a
formal run's ``proposals`` directory.  No simulator, GPU, model, or network
dependency is imported.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def module():
    spec = importlib.util.spec_from_file_location("q35n_isolation_diagnosis_v2", HERE / "isolation_diagnosis_v2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import v2 module")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run() -> dict:
    v2 = module()
    cases = []

    with tempfile.TemporaryDirectory(prefix="q35n_v2_cpu_") as tmp:
        root = Path(tmp)
        fixture_root = root / "fixtures"
        fixture_root.mkdir()
        family = {"family_id": "fixture", "house": "fixture", "split": "TEST",
                  "roles": {}, "compiler": {}, "histories": {"H_A": []},
                  "traces": {"H_A__C0": {}}, "content_root": "fixture/content",
                  "proposal": {}, "training_admission": "fixture-only"}
        path = fixture_root / "fixture" / "FAMILY.json"
        v2.publish_family_once(path, family)
        assert json.loads(path.read_text())["content_root"] == "fixture/content"
        try:
            v2.publish_family_once(path, family)
        except FileExistsError:
            pass
        else:
            raise AssertionError("second publication was accepted")
        cases.append("one_time_publish")

        partial = dict(family)
        partial.pop("content_root")
        try:
            v2.validate_family_for_publish(partial)
        except ValueError:
            pass
        else:
            raise AssertionError("partial family was accepted")
        cases.append("partial_recovery_rejected")

    assert v2.classify_exception(v2.Rejected("COLLISION")) == "PHYSICAL_REJECTED"
    assert v2.classify_exception(ValueError("program")) == "ERROR"
    assert v2.classify_exception(FileNotFoundError("evidence")) == "EVIDENCE_MISSING"
    cases.append("exception_separation")

    Compiler = v2.import_compiler()
    compiler = Compiler(
        roles={"anchor_A": ["chair", "room"], "anchor_B": ["bed", "room"],
               "terminal": ["plant", "room"]},
        tasks={"task_A": {"anchor": "anchor_A", "terminal": "terminal",
                           "instruction": "fixture"}},
        eligible={"anchor_A": [7], "anchor_B": [8], "terminal": [9]},
    )
    obs = [
        {"step": 0, "evidence_complete": True, "pixels": {"7": 300, "8": 300, "9": 0}},
        {"step": 1, "evidence_complete": True, "pixels": {"7": 256, "8": 255, "9": 0}},
        {"step": 2, "evidence_complete": True, "pixels": {"7": 255, "8": 300, "9": 300}},
    ]
    events = compiler.atoms(obs)
    assert events[1]["anchor_A"] == [7]
    assert events[1]["anchor_B"] == []
    assert events[2]["anchor_A"] == []
    assert events[2]["anchor_B"] == []  # frame 1 was below 256; no cross-frame merge
    cases.append("see2_same_instance_threshold")

    guard = v2.diagnosis_only_guard()
    assert guard["hits"] == {}
    assert guard["new_physical_candidate_executions"] == 0
    assert v2.HISTORY_CAP == 240 and v2.FULL_DECISION_CAP == 500
    cases.append("diagnosis_only_guard_and_budget")

    return {"status": "PASS", "test_suite": "test_isolation_diagnosis_v2.py",
            "cases": cases, "passed": len(cases), "failed": 0,
            "fixture_directory": "temporary fixtures/ (not a formal proposals directory)", "formal_family_files_created": 0,
            "new_physical_candidate_executions": 0, "gpu_used": False,
            "simulator_used": False, "model_loaded": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result = run()
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
