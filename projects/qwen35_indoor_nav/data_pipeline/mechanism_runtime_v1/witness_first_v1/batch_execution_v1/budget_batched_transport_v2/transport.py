"""Exact fresh-only clock batching transport: batch07 -> batch08 path only."""
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent;BE=HERE.parent;WF=BE.parent;RUNTIME=WF.parent
ORIGINAL=BE/'budget_batched_transport_v1/transport.py'
ORIGINAL_SHA='386e05e4ae16a1ed062846120fd5352ed4819594a8b4da4462a8645a2e9b1a2c'
def adapted_source(source):
    assert hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_SHA,'FROZEN_CLOCK_TRANSPORT_CHANGED'
    changes=[("BE/'batch_07'","BE/'batch_08'"),('EXACT_FRESH_BATCH07_ONLY','EXACT_FRESH_BATCH08_ONLY')]
    value=source
    for old,new in changes:assert value.count(old)==1;value=value.replace(old,new)
    reverse=value
    for old,new in reversed(changes):reverse=reverse.replace(new,old)
    assert reverse==source
    return value
private=types.ModuleType('fresh_clock_transport_v2_private');private.__file__=str(HERE/'transport.py')
exec(compile(adapted_source(ORIGINAL.read_text()),str(ORIGINAL)+'::batch08_path_only','exec'),private.__dict__)
sha=private.sha;load=private.load;base=private.base
CLOCK=private.CLOCK;CLOCK_SHA=private.CLOCK_SHA
check_config=private.check_config;check_inputs=private.check_inputs
bind_budget=private.bind_budget;build_worker=private.build_worker
worker_main=private.worker_main;run_main=private.run_main
