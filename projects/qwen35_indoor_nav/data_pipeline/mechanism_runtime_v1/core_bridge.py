"""Read-only verified reuse of sealed CPU factory; no source monkey patches."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parent.parent/'mechanism_factory_v2'


def load(name):
    expected = dict((row.split(None, 1)[1], row.split(None, 1)[0])
                    for row in (CORE/'SHA256SUMS').read_text().splitlines())
    path = CORE/(name+'.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name], path
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve() == path.resolve(), 'MODULE_NAME_COLLISION:'+name
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compiler = load('compiler')
planning = load('planning')
factory = load('factory')
Compiler, digest, ref = compiler.Compiler, compiler.digest, compiler.ref
slice_continuation = compiler.slice_continuation
FamilyFactory, TraceRunner, Reject = factory.FamilyFactory, factory.TraceRunner, factory.Reject
BudgetLedger, FreezeLedger, BudgetExceeded = planning.BudgetLedger, planning.FreezeLedger, planning.BudgetExceeded
