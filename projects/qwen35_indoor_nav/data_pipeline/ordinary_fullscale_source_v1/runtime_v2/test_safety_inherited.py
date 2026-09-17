"""Run the reviewed V1 safety behavior tests against transported new modules."""
from pathlib import Path
import sys
import types
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
import common
import run
import safe_size
import merge


def suite():
    text=transport.exact(transport.original('test_runtime.py'),
        'HERE=Path(__file__).resolve().parent',f'HERE=Path({str(HERE)!r})')
    text=transport.exact(text,
        '''    def test_restore_never_uses_tmux_kill_pane(self):
        self.assertNotIn("'respawn-pane','-k'",(HERE/'run.py').read_text())
        self.assertIn('os.kill(pid,signal.SIGTERM)',(HERE/'run.py').read_text())''',
        '''    def test_restore_never_uses_tmux_kill_pane(self):
        self.assertNotIn("'respawn-pane','-k'",transport.run_source())
        self.assertIn('os.kill(pid,signal.SIGTERM)',transport.run_source())''')
    module=types.ModuleType('inherited_safety_behavior');module.__file__=str(HERE/'test_safety_inherited.py')
    module.transport=transport
    exec(compile(text,module.__file__,'exec'),module.__dict__)
    selected=['test_exact_transport_rejects','test_safe_census_known_atoms','test_safe_census_errors_not_swallowed',
        'test_xml_includes_graphics','test_foreign_actual_task_rejected','test_alias_key_identity_source_scoped',
        'test_strict_auditor_unchanged','test_holder_exit_does_not_read_disappearing_cwd',
        'test_holder_exit_enoent_is_normal','test_holder_pid_reuse_is_not_exit',
        'test_pane_dead_transition_is_bounded_wait','test_restore_failure_preserves_remain_on',
        'test_restore_success_changes_remain_only_after_gpu_evidence',
        'test_residual_holder_context_drains_before_guard','test_real_new_external_is_not_ignored_during_drain',
        'test_restore_never_uses_tmux_kill_pane','test_merge_complete_alias_and_source_root',
        'test_merge_duplicate_physical_or_alias_rejected','test_merge_reaudit_mismatch_fails_closed']
    return unittest.TestSuite(module.RuntimeTests(name) for name in selected)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(not result.wasSuccessful())
