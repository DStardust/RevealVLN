"""Run blocking regressions against the immutable v2 implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
V16 = PACKAGE.parent
EXPECTED_REAL_SHA256 = "05cb829e2c1b383b2478a6859686618f85a9b98bb31098ed83cad9219ff80772"
REAL_PROPOSAL = "V16R1_18fe6790548a53d278cf"


def load_old_module():
    path = V16 / "isolation_diagnosis_v2.py"
    spec = importlib.util.spec_from_file_location("q35n_fixed_v2_pref_fix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"OLD_MODULE_IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def case(case_id: str, operation: Callable[[], None]) -> dict[str, object]:
    try:
        operation()
    except AssertionError as exc:
        return {"test_id": case_id, "status": "EXPECTED_FAIL_REPRODUCED", "failure": str(exc)}
    except BaseException as exc:
        return {"test_id": case_id, "status": "ERROR", "failure": repr(exc)}
    return {"test_id": case_id, "status": "UNEXPECTED_PASS"}


def run(real_source_root: Path) -> dict[str, object]:
    old = load_old_module()
    manifest = read_json(V16 / "DATA_MANIFEST.json")
    house = next(row for row in manifest["houses"] if row["house"] == "rqfALeAoiTq")
    proposals = read_json(V16 / "runs/v16_repair_rqf_001/PROPOSALS_rqfALeAoiTq.json")
    proposal = next(row for row in proposals if row["id"] == REAL_PROPOSAL)
    real_probe = real_source_root / (
        "projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/"
        f"runs/v16_repair_rqf_001/proposals/{REAL_PROPOSAL}/yaw_00/INITIAL_VIEW_PROBE.json"
    )
    probe_bytes = real_probe.read_bytes()
    actual_sha = hashlib.sha256(probe_bytes).hexdigest()
    if actual_sha != EXPECTED_REAL_SHA256:
        raise RuntimeError(f"REAL_REGRESSION_EVIDENCE_IDENTITY:{actual_sha}")

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="q35n_pref_fix_") as tmp_name:
        tmp = Path(tmp_name)

        def terminal_only() -> None:
            run_root = tmp / "terminal"
            run_root.mkdir()
            (run_root / "PROPOSALS_rqfALeAoiTq.json").write_text(
                json.dumps([proposal]), encoding="utf-8"
            )
            yaw = run_root / "proposals" / REAL_PROPOSAL / "yaw_00"
            yaw.mkdir(parents=True)
            (yaw / "INITIAL_VIEW_PROBE.json").write_bytes(probe_bytes)
            audit = old.neutral_start_audit(run_root, house)
            row = audit["yaw_records"][0]
            assert row["roles_triggered"] == ["terminal"], row
            assert row["classification"] == "VALID_NEUTRAL_START", (
                "old anchor gate rejects terminal-only evidence: " + row["classification"]
            )

        results.append(case("N12", terminal_only))

        def missing_eight() -> None:
            run_root = tmp / "missing_eight"
            run_root.mkdir()
            (run_root / "PROPOSALS_rqfALeAoiTq.json").write_text(
                json.dumps([proposal]), encoding="utf-8"
            )
            proposal_root = run_root / "proposals" / REAL_PROPOSAL
            for yaw in range(0, 24, 3):
                (proposal_root / f"yaw_{yaw:02d}").mkdir(parents=True)
            audit = old.neutral_start_audit(run_root, house)
            result = audit["proposal_rollup"][0]["proposal_classification"]
            assert result == "INCOMPLETE_EVIDENCE", (
                "old rollup maps eight missing probes to " + result
            )

        results.append(case("N09", missing_eight))

        weak_family = {
            "family_id": "weak",
            "house": "rqfALeAoiTq",
            "split": "TEST",
            "roles": {},
            "compiler": {},
            "histories": {"H_A": []},
            "traces": {"H_A__C0": {}},
            "content_root": "content",
            "proposal": {},
            "training_admission": "pending",
        }

        def weak_family_validation() -> None:
            try:
                old.validate_family_for_publish(weak_family)
            except ValueError:
                return
            raise AssertionError("old validator accepted one-history/one-trace shell")

        results.append(case("F01", weak_family_validation))

        def partial_publication() -> None:
            destination = tmp / "partial" / "FAMILY.json"
            destination.parent.mkdir()
            original_dump = old.json.dump

            def broken_dump(value, stream, **kwargs):
                stream.write("{")
                stream.flush()
                raise OSError("injected serialization/write failure")

            old.json.dump = broken_dump
            try:
                try:
                    old.publish_family_once(destination, weak_family)
                except OSError:
                    pass
            finally:
                old.json.dump = original_dump
            assert not destination.exists(), "old publisher exposed a partial final FAMILY.json"

        results.append(case("F11", partial_publication))

        def output_missing_classification() -> None:
            result = old.classify_exception(FileNotFoundError("output directory"))
            assert result == "ERROR", "old classifier lacks boundary context and returned " + result

        results.append(case("E01", output_missing_classification))

    statuses = {row["status"] for row in results}
    return {
        "status": "EXPECTED_FAILURES_REPRODUCED" if statuses == {"EXPECTED_FAIL_REPRODUCED"} else "PREF_FIX_RESULT_REQUIRES_REVIEW",
        "old_module": str((V16 / "isolation_diagnosis_v2.py").resolve()),
        "old_module_sha256": hashlib.sha256((V16 / "isolation_diagnosis_v2.py").read_bytes()).hexdigest(),
        "real_regression_evidence": str(real_probe),
        "real_regression_evidence_sha256": actual_sha,
        "cases": results,
        "executed": len(results),
        "expected_failures_reproduced": sum(row["status"] == "EXPECTED_FAIL_REPRODUCED" for row in results),
        "unexpected_passes": sum(row["status"] == "UNEXPECTED_PASS" for row in results),
        "errors": sum(row["status"] == "ERROR" for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-source-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.real_source_root.resolve())
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["unexpected_passes"] == 0 and result["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
