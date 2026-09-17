"""Same sealed sample index; only the source sampling plan changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'ordinary_expanded_v1/data.py'
EXPECTED='1c98ef6716500e69c4778e011d523db6c149c10e32d1a1d36b52aca924da74d9'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==EXPECTED
s=importlib.util.spec_from_file_location('r2r_sealed_data',BASE)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
for name in ('load_rows','load_sample_index','SampleStore','advance_epoch_boundary','inflection_weight','ACTIONS'):
    globals()[name]=getattr(parent,name)

def plan_epoch_batches(samples,max_tokens,seed,epoch,world_size):
    assert (len(samples),max_tokens,seed,epoch,world_size)==(2650347,6144,1209,0,3),'PLAN_ARGUMENTS'
    protocol=json.loads((HERE/'PROTOCOL_FILESTORE.json').read_text())
    path=HERE/'PLAN.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['sampling_plan_sha256'],'PLAN_CHANGED'
    plan=json.loads(path.read_text())
    assert len(plan)==3 and all(len(r)==8691 for r in plan),'PLAN_SHAPE'
    return plan
