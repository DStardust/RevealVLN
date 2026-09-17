"""Generated CPU journal fixtures only, confined beneath mechanism_factory_v2."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q35n_journal_fixture", BASE / "journal.py")
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="journal_cpu_fixture_", dir=BASE)
        self.root = Path(self.temporary.name) / "journal"
        self.config = {"candidate": ["a", "b"], "budget": 3}

    def tearDown(self):
        self.temporary.cleanup()

    def create(self):
        with j.Journal(self.root, self.config) as journal:
            journal.append("budget", {"actions": 1})
            journal.append("budget", {"actions": 2})

    def test_append_resume_latest(self):
        self.create()
        with j.Journal(self.root, self.config, resume=True) as journal:
            self.assertEqual(journal.latest("budget"), {"actions": 2})
            self.assertIsNone(journal.latest("missing"))
            journal.append("budget", {"actions": 3})
        with j.Journal(self.root, self.config, resume=True) as journal:
            self.assertEqual(journal.latest("budget"), {"actions": 3})

    def test_no_implicit_resume_or_missing_resume(self):
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)
        self.create()
        with self.assertRaises(FileExistsError):
            j.Journal(self.root, self.config)

    def test_config_hash_frozen(self):
        self.create()
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, {"budget": 4}, resume=True)

    def test_nonblocking_single_writer_lock(self):
        with j.Journal(self.root, self.config):
            with self.assertRaises(j.JournalError):
                j.Journal(self.root, self.config, resume=True)
        with j.Journal(self.root, self.config, resume=True):
            pass

    def test_partial_line_rejected_and_preserved(self):
        self.create()
        path = self.root / "events.jsonl"
        damaged = path.read_bytes()[:-9]
        path.write_bytes(damaged)
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)
        self.assertEqual(path.read_bytes(), damaged)

    def test_complete_tail_record_deletion_rejected(self):
        self.create()
        path = self.root / "events.jsonl"
        path.write_bytes(b"\n".join(path.read_bytes().splitlines()[:-1]) + b"\n")
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)

    def test_same_size_content_tampering_rejected(self):
        self.create()
        path = self.root / "events.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"actions":1', b'"actions":9'))
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)

    def test_commit_interruption_poisoned_and_resume_closed(self):
        journal = j.Journal(self.root, self.config)
        def fail(head):
            raise OSError("simulated interrupt after log fsync, before HEAD")
        journal._commit_head = fail
        with self.assertRaises(OSError):
            journal.append("budget", {"actions": 1})
        with self.assertRaises(j.JournalError):
            journal.append("budget", {"actions": 2})
        journal.close()
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)

    def test_leftover_head_candidate_rejected(self):
        self.create()
        (self.root / ".HEAD.next").write_bytes(b"partial")
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)

    def test_head_mismatch_rejected(self):
        self.create()
        path = self.root / "HEAD.json"
        head = json.loads(path.read_bytes())
        head["last_hash"] = "0" * 64
        path.write_bytes(j._canonical(head))
        with self.assertRaises(j.JournalError):
            j.Journal(self.root, self.config, resume=True)

    def test_copy_isolation(self):
        with j.Journal(self.root, self.config) as journal:
            payload = {"actions": [1]}
            record = journal.append("budget", payload)
            payload["actions"].append(2)
            record["payload"]["actions"].append(3)
            read = journal.latest("budget")
            read["actions"].append(4)
            self.assertEqual(journal.latest("budget"), {"actions": [1]})

    def test_scope_escape_rejected_before_create(self):
        with self.assertRaises(j.JournalError):
            j.Journal(BASE.parent / "forbidden_journal_fixture", self.config)

    def test_symlink_escape_rejected(self):
        alias = Path(self.temporary.name) / "alias"
        target = Path(self.temporary.name) / "target"
        target.mkdir()
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(j.JournalError):
            j.Journal(alias / "journal", self.config)

    def test_event_kind_and_nonfinite_rejected(self):
        with j.Journal(self.root, self.config) as journal:
            for kind in ("", "__config__", 2):
                with self.assertRaises(j.JournalError):
                    journal.append(kind, {})
            with self.assertRaises(ValueError):
                journal.append("budget", {"bad": float("nan")})
            journal.append("budget", {"actions": 1})

    def test_budget_callback_can_recover_latest(self):
        with j.Journal(self.root, self.config) as journal:
            persist = lambda state: journal.append("budget", state)
            persist({"total_reserved_actions": 4, "active": ["a", "discovery"]})
        with j.Journal(self.root, self.config, resume=True) as journal:
            self.assertEqual(journal.latest("budget")["total_reserved_actions"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
