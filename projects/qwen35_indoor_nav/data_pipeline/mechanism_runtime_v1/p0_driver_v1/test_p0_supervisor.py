"""CPU mock regressions: no GPU queries, subprocesses, or real signals.

Fixture files live only under this runtime and are removed after each test.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ACTUAL_RUNTIME = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("q35n_p0_supervisor_tested", Path(__file__).with_name("supervisor.py"))
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)
GUARD_SPEC = importlib.util.spec_from_file_location("q35n_p0_test_guard", ACTUAL_RUNTIME / "guard.py")
guard = importlib.util.module_from_spec(GUARD_SPEC)
GUARD_SPEC.loader.exec_module(guard)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p0_supervisor_test_", dir=ACTUAL_RUNTIME)
        self.base = Path(self.temp.name)
        self.runtime = self.base / "runtime"
        self.runtime.mkdir()
        (self.runtime / "p0_driver_v1").mkdir()
        (self.runtime / "smoke_v1").mkdir()
        self.out = self.runtime / "out"
        self.out.mkdir()
        self.env = self.base / "env"
        (self.env / "bin").mkdir(parents=True)
        self.python = self.env / "bin/python3"
        self.python.write_bytes(b"mock interpreter only; never executable")
        (self.runtime / "guard.py").write_bytes(b"# mock code lock fixture\n")
        (self.runtime / "p0_driver_v1/worker.py").write_bytes(b"# mock worker; never executed\n")
        write_json(self.runtime / "smoke_v1/result.json", {"runtime_pass": True})
        write_json(self.runtime / "smoke_v1/SUPERVISOR_RESULT.json",
                   {"returncode": 0, "error": None, "cleanup_complete": True})
        self.cfg = {"runtime_allowed": True, "phase": "five_fixed_bundle_constructability_P0",
                    "gpu_device": 2, "budget": {"total_actions": 200000, "total_seconds": 6000,
                    "discovery_actions": 20000, "discovery_seconds": 720,
                    "certification_actions": 20000, "certification_seconds": 480},
                    "disk_cap_bytes": 16*1024**3, "ram_cap_bytes": 16*1024**3,
                    "gpu_cap_mib": 8192, "candidates": [{"candidate_id": "MP5_%02d" % i} for i in range(5)],
                    "smoke_result_sha256": sha(self.runtime / "smoke_v1/result.json")}
        write_json(self.out / "EXECUTION_CONFIG.json", self.cfg)
        self.auth = {"approved": True, "P0_authorized": True,
                     "config_sha256": sha(self.out / "EXECUTION_CONFIG.json"),
                     "code_sha256": {str(p.relative_to(self.runtime)): sha(p)
                        for p in [self.runtime / "guard.py", self.runtime / "p0_driver_v1/worker.py"]},
                     "python_realpath": str(self.python), "python_sha256": sha(self.python)}
        write_json(self.out / "EXECUTION_AUTH.json", self.auth)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(supervisor, "HERE", self.runtime))
        self.stack.enter_context(patch.object(supervisor, "LINE", self.base))
        self.stack.enter_context(patch.object(supervisor, "ENV", self.env))
        self.stack.enter_context(patch.dict(sys.modules, {"guard": guard}))
        # Global process operations are denied even if an unexpected code path
        # reaches them. Specific successful mocks are installed by run_case.
        self.popen = self.stack.enter_context(patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("REAL_PROCESS_FORBIDDEN")))
        self.check_output = self.stack.enter_context(patch.object(supervisor.subprocess, "check_output", side_effect=AssertionError("REAL_COMMAND_FORBIDDEN")))
        self.killpg = self.stack.enter_context(patch.object(supervisor.os, "killpg", side_effect=AssertionError("REAL_SIGNAL_FORBIDDEN")))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def persist_auth(self):
        write_json(self.out / "EXECUTION_AUTH.json", self.auth)

    def assert_admission_rejected_without_process(self):
        with patch.object(supervisor, "gpu_state", side_effect=AssertionError("GPU_MUST_NOT_BE_QUERIED")) as gpu:
            with self.assertRaises(AssertionError):
                supervisor.run(self.out)
            gpu.assert_not_called()
        self.popen.assert_not_called()
        self.killpg.assert_not_called()

    def run_case(self, *, gpu_effect=None, cleanup_effect=None, disk_effect=None,
                 worker_returncode=0, start_error=None, ps_text=None):
        clean = {"index": 2, "uuid": "GPU-be1b30d0-517b-b079-871b-de195d35a1a2",
                 "memory_mib": 0, "utilization": 0, "processes": []}
        proc = Mock(pid=876543, returncode=worker_returncode)
        proc.poll.return_value = worker_returncode  # exits before first loop
        self.popen.side_effect = start_error
        self.popen.return_value = proc
        self.check_output.side_effect = None
        self.check_output.return_value = ps_text or f"{os.getpid()} 128\n876543 256\n"
        with contextlib.ExitStack() as local:
            gpu = local.enter_context(patch.object(supervisor, "gpu_state", side_effect=gpu_effect or [dict(clean), dict(clean), dict(clean)]))
            local.enter_context(patch.object(supervisor, "owned_session_members", return_value=[]))
            cleanup = local.enter_context(patch.object(supervisor, "stop_owned", side_effect=cleanup_effect, return_value=[]))
            local.enter_context(patch.object(supervisor, "disk_bytes", side_effect=disk_effect or [100, 100]))
            rc = supervisor.run(self.out)
        self.killpg.assert_not_called()
        return rc, gpu.call_count, cleanup.call_count, json.loads((self.out / "SUPERVISOR_RESULT.json").read_text())

    def test_valid_admission(self):
        supervisor.verify_admission(self.out, self.cfg, self.auth)

    def test_p0_permission_denied_before_any_process(self):
        self.auth["P0_authorized"] = False
        self.persist_auth()
        self.assert_admission_rejected_without_process()

    def test_config_hash_denied_before_any_process(self):
        self.auth["config_sha256"] = "0"*64
        self.persist_auth()
        self.assert_admission_rejected_without_process()

    def test_incomplete_code_lock_denied_before_any_process(self):
        self.auth["code_sha256"].pop("guard.py")
        self.persist_auth()
        self.assert_admission_rejected_without_process()

    def test_changed_code_denied_before_any_process(self):
        (self.runtime / "guard.py").write_bytes(b"# changed\n")
        self.assert_admission_rejected_without_process()

    def test_smoke_cleanup_failure_denied_before_any_process(self):
        write_json(self.runtime / "smoke_v1/SUPERVISOR_RESULT.json",
                   {"returncode": 0, "error": None, "cleanup_complete": False})
        self.assert_admission_rejected_without_process()

    def test_worker_exit_still_gets_terminal_sample(self):
        rc, gpu_calls, cleanup_calls, result = self.run_case()
        self.assertEqual((rc, gpu_calls, cleanup_calls), (0, 3, 1))
        self.assertTrue(result["cleanup_complete"])
        samples = json.loads((self.out / "RESOURCE_SAMPLES.json").read_text())
        self.assertEqual(len(samples), 1)
        self.assertIn(os.getpid(), samples[0]["own_tree_pids"])

    def test_terminal_gpu_limit_is_resource_censored(self):
        clean = {"index": 2, "uuid": "GPU-be1b30d0-517b-b079-871b-de195d35a1a2",
                 "memory_mib": 0, "utilization": 0, "processes": []}
        rc, _, _, result = self.run_case(gpu_effect=[clean, {**clean, "memory_mib": 8192}, clean])
        self.assertEqual(rc, 1)
        self.assertTrue(result["error"]["resource_censored"])

    def test_terminal_disk_highwater_is_resource_censored(self):
        rc, _, _, result = self.run_case(disk_effect=[14*1024**3+1, 100])
        self.assertEqual(rc, 1)
        self.assertTrue(result["error"]["resource_censored"])

    def test_final_absolute_disk_cap_is_resource_censored(self):
        rc, _, _, result = self.run_case(disk_effect=[100, 16*1024**3+1])
        self.assertEqual(rc, 1)
        self.assertTrue(result["error"]["resource_censored"])

    def test_cleanup_exception_cannot_pass_and_keeps_records(self):
        rc, _, _, result = self.run_case(cleanup_effect=RuntimeError("mock cleanup failed"))
        self.assertEqual(rc, 1)
        self.assertFalse(result["cleanup_complete"])
        self.assertEqual(result["cleanup_errors"][0]["stage"], "own_session_cleanup")
        self.assertTrue((self.out / "GPU_RESTORE_RESULT.json").exists())

    def test_final_gpu_query_exception_cannot_pass(self):
        clean = {"index": 2, "uuid": "GPU-be1b30d0-517b-b079-871b-de195d35a1a2",
                 "memory_mib": 0, "utilization": 0, "processes": []}
        rc, _, _, result = self.run_case(gpu_effect=[clean, clean, RuntimeError("mock GPU query failed")])
        self.assertEqual(rc, 1)
        self.assertFalse(result["cleanup_complete"])
        self.assertEqual(json.loads((self.out / "GPU_LEASE_AFTER.json").read_text()), {"state": "UNKNOWN"})

    def test_nonzero_worker_exit_cannot_pass(self):
        rc, _, _, result = self.run_case(worker_returncode=7)
        self.assertEqual(rc, 1)
        self.assertEqual(result["returncode"], 7)

    def test_start_failure_no_cleanup_of_unowned_process(self):
        clean = {"index": 2, "uuid": "GPU-be1b30d0-517b-b079-871b-de195d35a1a2",
                 "memory_mib": 0, "utilization": 0, "processes": []}
        rc, _, cleanup_calls, result = self.run_case(start_error=OSError("mock start failure"), gpu_effect=[clean, clean], disk_effect=[100])
        self.assertEqual(rc, 1)
        self.assertEqual(cleanup_calls, 0)
        self.assertIsNone(result["returncode"])

    def test_terminal_ram_includes_supervisor(self):
        rc, _, _, result = self.run_case(ps_text=f"{os.getpid()} {17*1024**2}\n876543 1\n")
        self.assertEqual(rc, 1)
        self.assertTrue(result["error"]["resource_censored"])


if __name__ == "__main__":
    unittest.main()
