"""Correct one stale checkpoint-index binding; scientific code/inputs unchanged."""
import ast,copy,hashlib,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_dev_v6r1'
TRAIN=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6r1'
PRIOR=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1'
REVIEW=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_EVAL_R2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (HERE/'SOURCE_LOCK.json').exists()
    import torch
    assert not torch.cuda.is_initialized()
    failed=read(OLD/'run_001/LAUNCH_RESULT.json')
    assert failed['status']=='SERVICE_FAILED' and failed['returncode']==1
    assert not (OLD/'run_001/MODEL_LOADED.json').exists() and not (OLD/'run_001/PROGRESS.json').exists()
    assert 'line 37, in main' in (OLD/'run_001/worker.log').read_text()
    assert failed['cleanup']['remaining']==[] and failed['foreign_processes_signaled']==[]
    assert read(PRIOR/'FINAL_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    original=read(OLD/'PROTOCOL.json');tp=read(TRAIN/'PROTOCOL_FILESTORE.json')
    state=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
    assert sha(original['checkpoint'])==original['checkpoint_sha256']=='d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a'
    assert state['binding']['protocol_sha256']==original['training_protocol_sha256']==sha(TRAIN/'PROTOCOL_FILESTORE.json')
    assert original['sample_index_sha256']!=state['binding']['sample_index_sha256']
    p=copy.deepcopy(original);p['sample_index_sha256']=tp['sample_index_sha256']
    assert {k for k in p if p[k]!=original[k]}=={'sample_index_sha256'}
    assert state['binding']['sample_index_sha256']==p['sample_index_sha256']
    assert state['cursor']['updates']==p['checkpoint_updates']==1000
    assert state['global_decisions']==state['charged_compute_decisions']==99047
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    assert all(int(s['step'])==5000 for s in state['optimizer']['state'].values())
    for path,d in read(OLD/'SOURCE_LOCK.json')['files'].items():assert sha(path)==d,path
    for n in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py','reuse.py','tree_size.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes(),n
    for n in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    assert sha(HERE/'EPISODES_PRIVILEGED.json')=='2f41a84a575aabe3ddbb5d9d41e29bd9c779eff93b673b80d4c06db0ba953b16'
    save(HERE/'PROTOCOL.json',p)
    loaded=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_LAUNCHER_IMPORT')
    assert loaded['OUT']==HERE/'run_001';size,miss=loaded['_scan'].tree_size(HERE);assert size>0 and miss==0
    for f in list(HERE.glob('*.py'))+list(REVIEW.glob('*.py')):ast.parse(f.read_text())
    save(HERE/'CPU_BINDING_ACCEPTANCE.json',dict(status='PASS',unix=time.time(),all_pre_model_checkpoint_assertions_verified=True,
      sample_index_binding_correct=True,only_protocol_field_changed='sample_index_sha256',
      same_model_weights=True,same100_inputs=True,same_policy_simulator_metrics=True,gpu_actions=0))
    save(HERE/'TRANSPORT_REVISION.json',dict(id='Q35N_ONPOLICY_EVAL_BINDING_R2',unix=time.time(),predecessor=str(OLD),
      predecessor_seconds=failed['wall_seconds'],predecessor_navigation_actions=0,predecessor_parameter_updates=0,
      original_sample_index_sha256=original['sample_index_sha256'],correct_sample_index_sha256=p['sample_index_sha256'],
      old_failure_preserved=True,no_training=True,automatic_retry=False))
    files=dict(read(OLD/'SOURCE_LOCK.json')['files'])
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[OLD/'run_001/LAUNCH_RESULT.json',
      OLD/'run_001/worker.log',PRIOR/'WORKFLOW_RESULT.json',PRIOR/'FINAL_CHECKPOINT_ACCEPTANCE.json']:
        files[str(f)]=sha(f)
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='one checkpoint source-binding correction; same fixed final and100evaluation inputs'))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_ONE_SOURCE_BINDING_REPAIR',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),optimizer_updates=0))
    assert not torch.cuda.is_initialized();print('CPU_FULL_CHECKPOINT_BINDING_AND_LAUNCHER_PASS')
if __name__=='__main__':main()
