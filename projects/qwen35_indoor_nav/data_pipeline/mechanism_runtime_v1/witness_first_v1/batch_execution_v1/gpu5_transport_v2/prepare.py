"""GPU5-only, winding-only derived preparation. CPU; no GPU/process operations."""
import importlib.util
import json
from pathlib import Path
import re
import types

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu5_prepare_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def adapted_prepare_source(source):
    changes=[
      ("assert gpu in (1,2) and variant in ('lr_v2','winding_v1')", "assert gpu == 5 and variant == 'winding_v1'"),
      ("gpu_uuid={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}[gpu]",
       "gpu_uuid={5:'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b'}[gpu]"),
      ("save(out/'EXECUTION_CONFIG.json',cfg)",
       "cfg.update(GPU5_TRANSPORT_METADATA)\n    save(out/'EXECUTION_CONFIG.json',cfg)"),
      ("for p in code:lock[str(p.resolve())]=sha(p)",
       "code += GPU5_TRANSPORT_LOCK_PATHS\n    for p in code:lock[str(p.resolve())]=sha(p)")]
    for old,new in changes:
        t.require(source.count(old)==1,'PREPARE_EXACT_SUBSTITUTION:'+old)
        source=source.replace(old,new)
    return source

WORKER='''import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu5_frozen_batch_worker',HERE.parent/'shared.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':module.worker_main(HERE)
'''
LAUNCHER='''import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu5_frozen_transport',HERE.parent/'gpu5_transport_v2/transport.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':module.run_main(HERE)
'''

def prepare(snapshot,name,indices,authorization,holder_identity):
    t.verify_sources()
    t.require(re.fullmatch(r'batch_[0-9]+(?:r[0-9]+)?',name) is not None,'FRESH_BATCH_NAME')
    auth=Path(authorization).resolve(strict=True);identity=Path(holder_identity).resolve(strict=True)
    t.require(auth.is_relative_to(t.LINE/'authorizations') and identity.is_relative_to(t.LINE/'authorizations'),'AUTHORIZATION_SCOPE')
    t.validate_authorization(t.read(auth),t.BE/name,Path(snapshot).resolve(strict=True),indices,identity)
    t.normalize_identity(t.read(identity))
    t.require(not (t.BE/name).exists(),'NEW_BATCH_ONLY')
    t.require(1<=len(indices)<=3 and len(set(indices))==len(indices) and all(type(i) is int and i>=0 for i in indices),'SELECTION_INDICES')
    source=adapted_prepare_source((t.BE/'prepare.py').read_text())
    module=types.ModuleType('sealed_gpu5_prepare');module.__file__=str(t.BE/'prepare.py')
    exec(compile(source,str(HERE/'prepare.py'),'exec'),module.__dict__)
    module.GPU5_TRANSPORT_METADATA={'gpu5_transport_version':'gpu5_transport_v2',
        'gpu5_holder_identity_path':str(identity),'gpu5_authorization_path':str(auth),
        'gpu5_borrow_requires_separate_main_agent_approval':True}
    module.GPU5_TRANSPORT_LOCK_PATHS=[*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',auth,identity,
        t.ROOT/'scripts/occupy_idle_gpu.py',t.LINE/'data_pipeline/ordinary_scale_v1/recovery_v4/run.py',
        t.BE/'BATCH02_MAIN_RECOVERY_RECEIPT_V1.json',t.BE/'gpu5_transport_v1/SHA256SUMS']
    batch=t.BE/name;batch.mkdir(exist_ok=False)
    with (batch/'worker.py').open('x') as f:f.write(WORKER)
    with (batch/'run.py').open('x') as f:f.write(LAUNCHER)
    module.prepare(snapshot,name,indices,5,'winding_v1')
    cfg=t.read(batch/'run_v1/EXECUTION_CONFIG.json');lock=t.read(batch/'run_v1/INPUT_LOCK.json')
    t.require(cfg['gpu_device']==5 and cfg['gpu_uuid']==t.UUID and cfg['factory_variant']=='winding_v1','PREPARED_TRANSPORT_IDENTITY')
    for p in module.GPU5_TRANSPORT_LOCK_PATHS:t.require(lock[str(p.resolve())]==t.sha(p.resolve()),'TRANSPORT_CLOSURE')
    print(json.dumps({'prepared':True,'gpu':5,'runtime_started':False,
        'required_approval_path':str(batch/'MAIN_AGENT_GPU5_APPROVAL.json'),
        'required_approval':{'approved':True,'gpu':5,'input_lock_sha256':t.sha(batch/'run_v1/INPUT_LOCK.json'),
            'holder_identity_sha256':t.sha(identity)}}))
    return batch

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--indices',type=int,nargs='+',required=True)
    parser.add_argument('--authorization',required=True);parser.add_argument('--holder-identity',required=True)
    args=parser.parse_args();prepare(args.snapshot,args.name,args.indices,args.authorization,args.holder_identity)
