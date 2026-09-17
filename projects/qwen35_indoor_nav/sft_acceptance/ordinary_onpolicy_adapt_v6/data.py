"""Original loader plus a strict whitelist loader for audited visited-state advice."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
BASE=HERE.parent/'ordinary_expanded_v1/data.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='1c98ef6716500e69c4778e011d523db6c149c10e32d1a1d36b52aca924da74d9'
s=importlib.util.spec_from_file_location('onpolicy_original_data',BASE);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
load_sample_index=parent.load_sample_index;ACTIONS=parent.ACTIONS
advance_epoch_boundary=parent.advance_epoch_boundary
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load_rows():
    protocol=read(HERE/'PROTOCOL_FILESTORE.json')
    assert sha(HERE/'TRAINING_ROWS.json')==protocol['snapshot']['training_index_sha256']
    rows=read(HERE/'TRAINING_ROWS.json')
    old,report=parent.load_rows()
    assert rows[:37114]==old and len(rows)==37114+protocol['recovery_unique_inputs']
    for p,d in read(DATA/'DATA_SEAL.json')['files'].items():assert sha(p)==d,p
    assert read(DATA/'RESULT.json')['data_gate']
    assert sum(r['decisions'] for r in rows)==protocol['snapshot']['decisions_per_epoch']
    assert all(r['split']=='FIT' for r in rows)
    return rows,dict(training_index_sha256=sha(HERE/'TRAINING_ROWS.json'),records=len(rows))
def plan_epoch_batches(samples,max_tokens,seed,epoch,world_size):
    p=read(HERE/'PROTOCOL_FILESTORE.json')
    assert (len(samples),max_tokens,seed,epoch,world_size)==(p['snapshot']['decisions_per_epoch'],6144,1209,0,3)
    assert sha(HERE/'PLAN.json')==p['sampling_plan_sha256']
    plan=read(HERE/'PLAN.json');assert len(plan)==3 and all(len(r)==1000 for r in plan)
    return plan
class SampleStore:
    def __init__(self,rows):
        self.rows=rows;self.old=parent.SampleStore(rows[:37114])
        self.inputs={x['record_id']:x for x in map(json.loads,(DATA/'POLICY_INPUTS.jsonl').read_text().splitlines())}
    def get(self,record_idx,t):
        if record_idx<37114:return self.old.get(record_idx,t)
        from PIL import Image
        row=self.rows[record_idx];assert t==0 and row['source']=='AUDITED_ONPOLICY_GOAL_TEACHER'
        value=self.inputs[row['advice_record_id']]
        assert set(value)=={'record_id','instruction','rgb_sha256','executed_actions'}
        assert 1<=len(value['rgb_sha256'])<=2 and len(value['executed_actions'])<=8
        assert all(a in ACTIONS[:3] for a in value['executed_actions'])
        images=[]
        for digest in value['rgb_sha256']:
            path=(DATA/'content'/(digest+'.png')).resolve(strict=True)
            assert path.is_relative_to(DATA/'content')
            with Image.open(path) as im:
                assert im.mode=='RGB' and im.size==(224,224)
                assert hashlib.sha256(im.tobytes()).hexdigest()==digest
                images.append(im.copy())
        return dict(instruction=value['instruction'],images=images,executed=value['executed_actions'],target=row['target'])
