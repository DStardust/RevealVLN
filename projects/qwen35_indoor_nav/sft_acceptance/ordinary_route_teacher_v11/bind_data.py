"""Immutable CPU data-only binding after data gate/build; not training admission."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
a=json.loads((HERE/'BUILD_AUDIT.json').read_text());assert a['status']=='PASS'
p=dict(snapshot=dict(path=str(HERE/'TRAINING_ROWS.json'),training_index_sha256=sha(HERE/'TRAINING_ROWS.json'),
       decisions_per_epoch=2650347+a['recovery_unique_inputs']),recovery_unique_inputs=a['recovery_unique_inputs'],
       sample_index_sha256=sha(HERE/'SAMPLE_INDEX.jsonl'),sampling_plan_sha256=sha(HERE/'PLAN.json'))
with (HERE/'DATA_BINDING.json').open('x') as f:json.dump(p,f,indent=2)
print('DATA_BINDING_CREATED_NO_TRAINING')
