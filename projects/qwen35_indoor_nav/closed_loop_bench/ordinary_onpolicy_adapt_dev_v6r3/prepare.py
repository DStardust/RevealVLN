"""Supply actual mandatory CPU receipt and execute exact read-only launcher prefix."""
import ast,hashlib,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_dev_v6r2'
PRIOR=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_EVAL_R2'
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
    assert not (OLD/'run_001').exists() and not (OLD/'gpu1_eval.lock').exists()
    assert not (OLD/'CPU_TEST_RESULT.json').exists()
    assert read(PRIOR/'WORKFLOW_RESULT.json')['status']=='FAILED_OR_BLOCKED'
    assert 'CPU_TEST_RESULT.json' in (PRIOR/'eval_launcher.log').read_text()
    for path,d in read(OLD/'SOURCE_LOCK.json')['files'].items():assert sha(path)==d,path
    for n in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py','reuse.py','tree_size.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes()
    for n in ('PROTOCOL.json','EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json','CPU_BINDING_ACCEPTANCE.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    p=read(HERE/'PROTOCOL.json');state=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
    assert sha(p['checkpoint'])==p['checkpoint_sha256']=='d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a'
    assert state['binding']==dict(protocol_sha256=p['training_protocol_sha256'],sample_index_sha256=p['sample_index_sha256'])
    assert state['cursor']['updates']==p['checkpoint_updates']==1000
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    launch=runpy.run_path(str(HERE/'launch.py'),run_name='ACTUAL_CPU_IMPORT')
    assert launch['OUT']==HERE/'run_001'
    size,miss=launch['_scan'].tree_size(HERE);assert size>0 and miss==0
    reuse=runpy.run_path(str(HERE/'reuse.py'),run_name='CPU_SOURCE')
    for n in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):ast.parse(reuse['source'](n))
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,actual_launcher_import_passed=True,
      actual_checkpoint_all_binding_assertions_passed=True,same_scientific_code_and_protocol=True,
      same100_inputs=True,tree_size_passed=True,gpu_actions=0))
    save(HERE/'TRANSPORT_REVISION.json',dict(id='Q35N_ONPOLICY_EVAL_RECEIPT_R3',unix=time.time(),
      predecessor=str(OLD),predecessor_gpu_actions=0,predecessor_gpu_worker_started=False,
      only_change='mandatory CPU_TEST_RESULT receipt emitted from actual checks',no_training=True,automatic_retry=False))
    files=dict(read(OLD/'SOURCE_LOCK.json')['files'])
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[PRIOR/'WORKFLOW_RESULT.json',PRIOR/'eval_launcher.log']:
        files[str(f)]=sha(f)
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='mandatory CPU receipt completion; same fixed scientific protocol'))
    # Execute the launcher's actual AST prefix, not a separately retyped approximation.
    tree=ast.parse(reuse['source']('launch.py'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    prefix=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lease' for t in node.targets):break
        prefix.append(node)
    assert len(prefix)==3
    env=dict(launch);exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'ACTUAL_LAUNCHER_READONLY_PREFIX','exec'),env)
    assert env['p']==p
    gpu=launch['gpu_snapshot']()[p['gpu']]
    assert gpu['uuid']==p['gpu_uuid'] and not gpu['contexts'] and gpu['memory_mib']<128
    training=launch['training_snapshot']()
    assert not (HERE/'run_001').exists() and not (HERE/'gpu1_eval.lock').exists()
    save(HERE/'FINAL_LAUNCHER_PREFLIGHT.json',dict(passed=True,unix=time.time(),actual_source_lock_protocol_cpu_gate_executed=True,
      gpu1_empty_checked=True,training_status_readable=True,no_gpu_spawn=True,no_output_created=True))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_ONE_CPU_RECEIPT_REPAIR',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
      final_preflight_sha256=sha(HERE/'FINAL_LAUNCHER_PREFLIGHT.json'),optimizer_updates=0))
    assert not torch.cuda.is_initialized();print('ACTUAL_LAUNCHER_PREFIX_AND_ALL_CHECKPOINT_GATES_PASS')
if __name__=='__main__':main()
