"""Pre-freeze identical inputs; bind only the declared final checkpoint after training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
TRAIN=LINE/'sft_acceptance/ordinary_expanded_low_lr_v3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def freeze():
    assert not (HERE/'PENDING_SEAL.json').exists(),'ALREADY_FROZEN'
    s=importlib.util.spec_from_file_location('continue_eval_test',HERE/'reuse.py')
    r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
    checks=[]
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
        text=r.source(name);ast.parse(text);assert text==r.parent.source(name)
        checks.append(name+':exact_matched_predecessor')
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/name).open('xb') as f:f.write((BASE/name).read_bytes())
        assert sha(HERE/name)==sha(BASE/name)
    assert read(BASE/'run_001/RESULT.json')['selected_batch_size']==1
    template=read(BASE/'PROTOCOL.json')
    template.update(id='Q35N_ORDINARY_EXPANDED_LOW_LR_DEV_V3',checkpoint_role='low_lr_8000_pure_model',
        checkpoint=str(TRAIN/'formal/attempt_001/checkpoint_000008000.pt'),checkpoint_sha256=None,
        checkpoint_updates=8000,training_protocol_sha256=sha(TRAIN/'PROTOCOL_FILESTORE.json'),
        main_criterion='SR delta >0, SPL delta >=0, nDTW delta >=-0.01; engineering development only',
        controller_enabled=False,checkpoint_selection='fixed final 8000 after bounded continuation, never best-of-checkpoints')
    assert template['seed']==read(TRAIN/'PROTOCOL_FILESTORE.json')['seed']==1209
    save(HERE/'PROTOCOL_TEMPLATE.json',template)
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,checks=checks,identical_input=True,batch_size=1,controller_enabled=False))
    files={str(p):sha(p) for p in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'SPEC_ZH.md',TRAIN/'PROTOCOL_FILESTORE.json']}
    save(HERE/'PENDING_SEAL.json',dict(status='AWAIT_FINAL_CHECKPOINT_BINDING',files=files))
    print('PREFROZEN: identical inputs and matched evaluation implementation')


def bind():
    assert not (HERE/'SOURCE_LOCK.json').exists(),'ALREADY_BOUND'
    for path,digest in read(HERE/'PENDING_SEAL.json')['files'].items():assert sha(path)==digest,path
    final=read(TRAIN/'formal/attempt_001/RESULT.json');lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
    assert final['status']=='STOPPED' and final['stop']==['BUDGET:max_updates'] and final['cursor']['updates']==8000
    assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None
    review=LINE/'reviews/Q35N_ORDINARY_LOW_LR_V3/FIRST_CHECKPOINT_ACCEPTANCE.json'
    assert read(review)['status']=='PASS'
    p=read(HERE/'PROTOCOL_TEMPLATE.json');ckpt=Path(p['checkpoint'])
    assert str(ckpt)==final['latest_checkpoint'];receipt=read(str(ckpt)+'.json')
    assert receipt['cursor']['updates']==8000 and sha(ckpt)==receipt['sha256']
    p['checkpoint_sha256']=receipt['sha256'];save(HERE/'PROTOCOL.json',p)
    files=dict(read(BASE/'SOURCE_LOCK.json')['files'])
    for path,digest in files.items():assert sha(path)==digest,path
    files.update(read(HERE/'PENDING_SEAL.json')['files'])
    for path in (HERE/'PROTOCOL.json',HERE/'PENDING_SEAL.json',TRAIN/'PROTOCOL_FILESTORE.json',ckpt,Path(str(ckpt)+'.json'),review):files[str(path)]=sha(path)
    for name,digest in read(TRAIN/'PROTOCOL_FILESTORE.json')['code_sha256'].items():files[str(TRAIN/name)]=digest
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='fixed 8000 vs fixed 4000 pure models, batch one',training_updates=0))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_BOUNDED_MATCHED_DEV',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
        model_sha256=receipt['sha256'],automatic_retry=False,scientific_gain_verified=False))
    print('BOUND: final 8000 checkpoint')


if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['bind'])
    freeze() if sys.argv[1]=='freeze' else bind()
