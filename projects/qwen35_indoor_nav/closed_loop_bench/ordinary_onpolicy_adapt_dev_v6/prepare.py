"""Preseal the same 100 inputs; bind only the fixed full-epoch checkpoint later."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
TRAIN=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6'
REVIEW=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def freeze():
    assert not (HERE/'PENDING_SEAL.json').exists()
    s=importlib.util.spec_from_file_location('full_epoch_eval_freeze',HERE/'reuse.py')
    r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
    checks=[]
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
        text=r.source(name);ast.parse(text)
        if name!='aggregate.py':assert text==r.parent.source(name)
        checks.append(name)
    assert 'stop_logit_bias' not in r.source('evaluate.py')
    assert 'batch_size=1  # Matched comparison' in r.source('evaluate.py')
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/name).open('xb') as f:f.write((BASE/name).read_bytes())
        assert sha(HERE/name)==sha(BASE/name)
    template=read(BASE/'PROTOCOL.json')
    template.update(id='Q35N_ORDINARY_ONPOLICY_ADAPT_DEV_V6',checkpoint_role='base4000_plus_onpolicy1000_optimizer5000_pure_model',
         checkpoint=str(TRAIN/'formal/attempt_001/checkpoint_000001000.pt'),checkpoint_sha256=None,
         checkpoint_updates=1000,training_protocol_sha256=sha(TRAIN/'PROTOCOL_FILESTORE.json'),
         main_criterion='SR delta >0, SPL delta >=0, nDTW delta >=-0.01 vs pure4000',
         controller_enabled=False,checkpoint_selection='fixed correction1000, optimizer5000; no intermediate selection')
    save(HERE/'PROTOCOL_TEMPLATE.json',template)
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,checks=checks,policy_simulator_metrics_unchanged=True,
         batch_size=1,only_scan_transport_corrected=True,controller_enabled=False))
    import runpy
    launch=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_IMPORT_TEST')
    assert launch['OUT']==HERE/'run_001' and callable(launch['_scan'].tree_size)
    size,misses=launch['_scan'].tree_size(HERE);assert size>0 and misses==0
    assert not (HERE/'run_001').exists()
    save(HERE/'LAUNCHER_IMPORT_ACCEPTANCE.json',dict(passed=True,zero_gpu_actions=True))
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[TRAIN/'PROTOCOL_FILESTORE.json',TRAIN/'PLAN_ZH.md']
    save(HERE/'PENDING_SEAL.json',dict(status='AWAIT_FIXED_ONPOLICY_1000',files={str(p):sha(p) for p in paths}))
    print('PREFROZEN_SAME_100_FIXED_EPOCH_FINAL')


def bind():
    assert not (HERE/'SOURCE_LOCK.json').exists()
    for path,digest in read(HERE/'PENDING_SEAL.json')['files'].items():assert sha(path)==digest,path
    final=read(TRAIN/'formal/attempt_001/RESULT.json');tp=read(TRAIN/'PROTOCOL_FILESTORE.json')
    lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
    assert final['status']=='EPOCHS_COMPLETED' and final['stop']==[] and final['cursor']==tp['expected_final_cursor']
    assert final['global_decisions']==final['charged_compute_decisions']==tp['expected_final_global_decisions']
    assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None
    assert read(REVIEW/'FIRST_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    assert read(REVIEW/'FINAL_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    p=read(HERE/'PROTOCOL_TEMPLATE.json');ckpt=Path(p['checkpoint']);receipt=read(Path(str(ckpt)+'.json'))
    assert str(ckpt)==final['latest_checkpoint'] and receipt['cursor']==dict(epoch=0,position=1000,updates=1000,decisions=33142)
    assert sha(ckpt)==receipt['sha256'];p['checkpoint_sha256']=receipt['sha256'];save(HERE/'PROTOCOL.json',p)
    files=dict(read(BASE/'SOURCE_LOCK.json')['files']);files.update(read(HERE/'PENDING_SEAL.json')['files'])
    for name,digest in tp['code_sha256'].items():files[str(TRAIN/name)]=digest
    extra=[HERE/'PROTOCOL.json',HERE/'PENDING_SEAL.json',ckpt,Path(str(ckpt)+'.json'),REVIEW/'FIRST_CHECKPOINT_ACCEPTANCE.json']
    transport=HERE.parent/'ordinary_stop_calibration_fit_v2'
    for path in extra+[transport/'reuse.py',transport/'tree_size.py',transport.parent/'ordinary_stop_calibration_fit_v1/reuse.py']:
        files[str(path)]=sha(path)
    for path,digest in files.items():assert sha(path)==digest,path
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='same100, fixed base4000 plus correction1000 optimizer5000',training_updates=0))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_FIXED_ONPOLICY_1000_DEV',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
         checkpoint_sha256=receipt['sha256'],automatic_retry=False,controller_enabled=False))
    print('BOUND_FIXED_EPOCH_END')


if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['bind'])
    freeze() if sys.argv[1]=='freeze' else bind()
