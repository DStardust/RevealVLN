import importlib.util
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import run
from runtime_journal import Journal

HERE = Path(__file__).resolve().parent


class SupervisorTests(unittest.TestCase):
    def test_owned_session_excludes_external_and_zombies(self):
        raw = '44 44 44 S\n45 44 44 S\n46 46 44 S\n47 47 47 S\n48 44 44 Z\n'
        with patch.object(run.subprocess, 'check_output', return_value=raw):
            self.assertEqual(run.owned_session_members(44), [(44, 44), (45, 44), (46, 46)])

    def test_parent_exit_does_not_skip_child_cleanup(self):
        proc = Mock(pid=44)
        proc.poll.return_value = 0
        with patch.object(run, 'owned_session_members', side_effect=[[(45, 44)], [(45, 44)], []]), \
                patch.object(run.os, 'killpg') as kill, patch.object(run.time, 'sleep'):
            self.assertEqual(run.stop_owned(proc), [])
            kill.assert_called_once_with(44, signal.SIGTERM)

    def test_already_finished_group_sends_no_signal(self):
        proc = Mock(pid=44)
        with patch.object(run, 'owned_session_members', return_value=[]), patch.object(run.os, 'killpg') as kill:
            self.assertEqual(run.stop_owned(proc), [])
            kill.assert_not_called()

    def test_runtime_scoped_journal_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='journal_test_') as directory:
            root = Path(directory)/'journal'
            with Journal(root, {'CPU_only': True}) as journal:
                journal.append('budget', {'actions': 4})
            with Journal(root, {'CPU_only': True}, resume=True) as journal:
                self.assertEqual(journal.latest('budget'), {'actions': 4})


if __name__ == '__main__':
    unittest.main()
