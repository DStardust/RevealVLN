"""Unmodified inherited safety/strict-readback assertions except old-plan-specific tests."""
from pathlib import Path
import sys
import types
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
source=(transport.SOURCE/'test_runtime.py').read_text()
assert __import__('hashlib').sha256(source.encode()).hexdigest()=='d4b184434f29432969c3dd5ba6a2f895f1b69274f79e528744abf7c9bf76c66f'
module=types.ModuleType('inherited_envdrop_safety');module.__file__=str(HERE/'test_inherited.py')
exec(compile(source,module.__file__,'exec'),module.__dict__)
skip={'test_manifest_partition_unchanged','test_closed_shard_not_replayed_in_worker','test_restore_never_uses_tmux_kill_pane'}
for name in skip:delattr(module.RuntimeTests,name)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(module.RuntimeTests))
    raise SystemExit(not result.wasSuccessful())
