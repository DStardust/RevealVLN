"""CPU-only single equal-parameter merge; prepare fixed FIT then gated DEV."""
import ast,hashlib,importlib.util,json,os,runpy,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
REVIEW=LINE/'reviews/Q35N_ORDINARY_EQUAL_MERGE_V7R1'
FIT=LINE/'closed_loop_bench/ordinary_equal_merge_fit_v7r1'
DEV=LINE/'closed_loop_bench/ordinary_equal_merge_dev_v7r1'
FITBASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
DEVBASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
A=LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt'
B=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6r1/formal/attempt_001/checkpoint_000001000.pt'
SHA_A='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
SHA_B='d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def activate(case):
    assert not (case/'SOURCE_LOCK.json').exists() and not (case/'run_001').exists()
    pending=read(case/'PENDING_SEAL.json')
    for p,d in pending['files'].items():assert sha(p)==d,p
    paths=dict(pending['files'])
    if case==DEV:
        gate=read(REVIEW/'FIT_GATE.json');assert gate['pass_gate'] and not gate['independent_validation']
        paths[str(REVIEW/'FIT_GATE.json')]=sha(REVIEW/'FIT_GATE.json')
        paths[str(FIT/'run_001/RESULT.json')]=sha(FIT/'run_001/RESULT.json')
    paths[str(case/'PENDING_SEAL.json')]=sha(case/'PENDING_SEAL.json')
    save(case/'SOURCE_LOCK.json',dict(files=paths,scope='single equal-parameter merge; no training; fixed scientific inputs'))
    # Execute original actual read-only launcher prefix, including mandatory receipt.
    launch=runpy.run_path(str(case/'launch.py'),run_name='CPU_ACTUAL_IMPORT')
    reuse=runpy.run_path(str(case/'reuse.py'),run_name='CPU_SOURCE')
    tree=ast.parse(reuse['source']('launch.py'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    prefix=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lease' for t in node.targets):break
        prefix.append(node)
    assert len(prefix)==3
    env=dict(launch);exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'ACTUAL_LAUNCHER_PREFIX','exec'),env)
    p=env['p'];gpu=launch['gpu_snapshot']()[1]
    assert gpu['uuid']==p['gpu_uuid'] and not gpu['contexts'] and gpu['memory_mib']<128
    launch['training_snapshot']()
    assert not (case/'run_001').exists() and not (case/'gpu1_eval.lock').exists()
    save(case/'MAIN_REVIEW.json',dict(status='PASS_FOR_ONE_FIXED_MERGE_'+('FIT' if case==FIT else 'DEV'),
      source_lock_sha256=sha(case/'SOURCE_LOCK.json'),actual_launcher_prefix_checked=True,requires_empty_gpu1=True,
      no_training=True,automatic_retry=False,fit_gate_required=(case==DEV),unix=time.time()))
def build():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (HERE/'MODEL_GENERATION_SPEC.json').exists()
    import torch
    assert not torch.cuda.is_initialized()
    assert sha(A)==SHA_A and sha(B)==SHA_B
    a=torch.load(A,map_location='cpu',weights_only=True);b=torch.load(B,map_location='cpu',weights_only=True)
    assert set(a['trainable'])==set(b['trainable']) and len(a['trainable'])==28
    assert a['cursor']['updates']==4000 and b['cursor']['updates']==1000
    assert all(int(x['step'])==4000 for x in a['optimizer']['state'].values())
    assert all(int(x['step'])==5000 for x in b['optimizer']['state'].values())
    spec=dict(id='Q35N_EQUAL_PARAMETER_MERGE_V7',alpha=.5,allowed_alphas=[.5],inference_only=True,
      parameter_updates=0,optimizer_resumption_allowed=False,merge='same_named_trainable_parameter_arithmetic_mean',arithmetic='FP32 then cast once to each original dtype',
      predecessor_cpu_failure_sha256=sha(HERE.parent/'ordinary_equal_merge_v7/BUILD_FAILURE.json'),
      not_logit_or_effective_LoRA_matrix_ensemble=True,plan_sha256=sha(HERE/'PLAN_ZH.md'),build_code_sha256=sha(Path(__file__)),
      source_a=dict(path=str(A),sha256=SHA_A,cursor=a['cursor'],optimizer_step=4000),
      source_b=dict(path=str(B),sha256=SHA_B,cursor=b['cursor'],optimizer_step=5000),
      actual_expanded_union_sample_index_sha256=b['binding']['sample_index_sha256'],
      known_foundation='https://github.com/mlfoundations/wise-ft/blob/master/src/wise_ft.py',
      novelty_claim=False)
    save(HERE/'MODEL_GENERATION_SPEC.json',spec)
    weights={};stats={}
    for k,x in a['trainable'].items():
        y=b['trainable'][k];assert x.shape==y.shape and x.dtype==y.dtype and x.dtype in (torch.float32,torch.bfloat16)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        z=(x.float()*.5+y.float()*.5).to(x.dtype);assert torch.isfinite(z).all()
        assert torch.equal(x*(1.-0.)+y*0.,x) and torch.equal(x*(1.-1.)+y*1.,y)
        assert torch.equal(z,(y.float()*.5+x.float()*.5).to(x.dtype))
        weights[k]=z
        stats[k]=dict(shape=list(x.shape),dtype=str(x.dtype),rounding_max_abs=float((z.double()-(x.double()+y.double())*.5).abs().max()),source_delta_l2=float(torch.linalg.vector_norm(y.double()-x.double())),
          merged_delta_l2=float(torch.linalg.vector_norm(z.double()-x.double())))
    state=dict(trainable=weights,binding=dict(protocol_sha256=sha(HERE/'MODEL_GENERATION_SPEC.json'),
      sample_index_sha256=b['binding']['sample_index_sha256']),cursor=dict(epoch=0,position=0,updates=0,decisions=0),
      inference_only=True,parameter_updates=0,source_checkpoints=[spec['source_a'],spec['source_b']],
      optimizer_resume_allowed=False)
    ckpt=HERE/'merged_equal_inference_only.pt'
    with ckpt.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    back=torch.load(ckpt,map_location='cpu',weights_only=True)
    assert set(back['trainable'])==set(weights) and 'optimizer' not in back
    assert all(torch.equal(v,back['trainable'][k]) for k,v in weights.items())
    save(HERE/(ckpt.name+'.json'),dict(sha256=sha(ckpt),inference_only=True,parameter_updates=0,optimizer_resume_allowed=False,
      cursor=state['cursor'],sources=[spec['source_a'],spec['source_b']]))
    save(HERE/'CPU_MERGE_ACCEPTANCE.json',dict(status='PASS',unix=time.time(),parameter_tensor_count=28,
      total_parameters=sum(v.numel() for v in weights.values()),all_state_finite=True,roundtrip_exact=True,
      alpha0_and1_endpoints_exact=True,equal_mean_order_invariant=True,alpha=.5,parameter_updates=0,
      model_architecture_unchanged=True,one_forward_per_action=True,tensor_differences=stats))
    common_files={str(f):sha(f) for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+
      [HERE/'PLAN_ZH.md',ckpt,HERE.parent/'ordinary_equal_merge_v7/BUILD_FAILURE.json',HERE.parent/'ordinary_equal_merge_v7/MODEL_GENERATION_SPEC.json']+list(REVIEW.glob('*.py'))+[A,B]}
    for case,base in ((FIT,FITBASE),(DEV,DEVBASE)):
        for n in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
            with (case/n).open('xb') as f:f.write((base/n).read_bytes())
            assert sha(case/n)==sha(base/n)
        p=read(base/'PROTOCOL.json')
        p.update(id='Q35N_EQUAL_PARAMETER_MERGE_'+('FIT' if case==FIT else 'DEV')+'_V7',
          checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),training_protocol_sha256=state['binding']['protocol_sha256'],
          sample_index_sha256=state['binding']['sample_index_sha256'],checkpoint_updates=0,
          checkpoint_role='inference_only_equal_mean_base4k_and_corrected1k',checkpoint_selection='single fixed alpha0.5 no sweep',
          main_criterion='FIT: SR>baseline,SPL>=,nDTW delta>=-.01,losses<=2; DEV requires FIT gate and original SR/SPL/nDTW gate',
          parameter_generation='CPU arithmetic; training_protocol_sha256 field binds MODEL_GENERATION_SPEC, not a training run',
          no_optimizer_state=True,threshold_fitting_allowed=False,controller_enabled=False)
        save(case/'PROTOCOL.json',p)
        source=runpy.run_path(str(case/'reuse.py'),run_name='CPU_SOURCE')
        for n in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
            ast.parse(source['source'](n))
            if n!='aggregate.py':assert source['source'](n)==source['parent'].source(n)
        launch=runpy.run_path(str(case/'launch.py'),run_name='CPU_IMPORT')
        size,miss=launch['_scan'].tree_size(case);assert size>0 and miss==0
        assert back['binding']==dict(protocol_sha256=p['training_protocol_sha256'],sample_index_sha256=p['sample_index_sha256'])
        assert back['cursor']['updates']==p['checkpoint_updates']
        save(case/'CPU_TEST_RESULT.json',dict(passed=True,actual_launcher_import=True,checkpoint_binding_exact=True,
          tree_size_passed=True,scientific_code_unchanged=True,inputs_byte_identical=True,gpu_actions=0))
        files=dict(read(base/'SOURCE_LOCK.json')['files']);files.update(common_files)
        baseline_paths=[base/'run_001/RESULT.json']
        for folder in (base/'run_001/lanes').glob('lane_*'):
            baseline_paths+=list(folder.glob('episode_*.json'))+[folder/n for n in ('INTERFACE.jsonl','POLICY_STEPS.jsonl','STEPS_PRIVILEGED.jsonl')]
        for f in list(case.glob('*.py'))+list(case.glob('*.json'))+baseline_paths:files[str(f)]=sha(f)
        for path,d in files.items():assert sha(path)==d,path
        save(case/'PENDING_SEAL.json',dict(files=files,fit_gate_required=(case==DEV),one_candidate=True))
    assert sha(A)==SHA_A and sha(B)==SHA_B and not torch.cuda.is_initialized()
    activate(FIT)
    print('ONE_EQUAL_MERGE_READY_FIT_ADMITTED_DEV_LOCKED_PENDING_GATE')
if __name__=='__main__':
    assert sys.argv[1:] in (['build'],['activate_dev'])
    build() if sys.argv[1]=='build' else activate(DEV)
