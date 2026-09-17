"""CPU-only strict V3 GPU6 merge; V4 owns GPU7 shards3/5 separately.

No final merge on import. No original runtime/source/receipt is modified.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
ROOT=HERE.parents[4]
V3=FULL/'runtime_v3'
V4=FULL/'runtime_v4'
V3_SHA='17de3b1d3377f1b2f4a13175d1a273c4714561a9d3a99399addbb166b4991684'
V4_SHA='a10c94789002cd6373d5522595935cea9bbebceb082cb3242272657f8a107669'


def sha(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return json.loads(path.read_text())


def validate_handoff(handoff,old,config,approval):
    assert handoff['status']=='PREWORKER_ONLY_EXCLUSIVE_OWNERSHIP_TRANSFER'
    assert handoff['old_runtime']=='runtime_v3' and handoff['new_runtime']=='runtime_v4'
    assert handoff['gpu']==7 and handoff['shards']==[3,5] and handoff['old_failure_preserved']
    assert old['workers']==[],'OLD_GPU7_HAD_ACTUAL_WORKERS'
    assert old['error']=="AssertionError('UNEXPECTED_NEW_SLEEPER_COMMAND')"
    assert old['restoration']['restored'] is False,'OLD_FAILED_RESTORE_MUST_REMAIN_UNCHANGED'
    assert config['gpu_to_shards']=={'7':[3,5]} and config['selected_shards']==[3,5]
    assert approval['approved'] is True and approval['gpu']==7 and approval['shards']==[3,5]
    assert approval['input_lock_sha256']==V4_SHA
    assert set(handoff['roots'])=={'3','5'}
    for s in (3,5):
        root=handoff['roots'][str(s)]
        assert root['root_input_lock_sha256']==V3_SHA and root['route_outputs_present'] is False
        assert root['output_root']==str((FULL/'production'/f'shard_{s:04d}').relative_to(ROOT))


def ownership_evidence():
    for runtime,expected in ((V3,V3_SHA),(V4,V4_SHA)):
        assert sha(runtime/'INPUT_LOCK.json')==expected
        for path,h in read(runtime/'INPUT_LOCK.json').items():assert sha(ROOT/path)==h,('SEALED_INPUT_CHANGED',path)
    handoff=read(V4/'PREWORKER_OWNERSHIP.json')
    old_attempt=V3/'lanes/gpu_7/attempt_000'
    old=read(old_attempt/'RESULT.json')
    config=read(V4/'PREPARED_CONFIG.json');approval=read(V4/'MAIN_AGENT_APPROVAL_GPU7.json')
    validate_handoff(handoff,old,config,approval)
    assert old['restoration']==read(old_attempt/'RESTORATION.json')
    expected={str((old_attempt/name).relative_to(ROOT)) for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json')}
    expected.add(str((V3/'INPUT_LOCK.json').relative_to(ROOT)))
    assert set(handoff['old_receipt_sha256'])==expected
    for p,h in handoff['old_receipt_sha256'].items():assert sha(ROOT/p)==h,'OLD_PREWORKER_RECEIPT_CHANGED'
    ids_path=ROOT/read(V4/'SETUP.json')['identities_path']
    assert approval['identity_sha256']==sha(ids_path)
    files=[V3/'INPUT_LOCK.json',V4/'INPUT_LOCK.json',V4/'PREWORKER_OWNERSHIP.json',V4/'PREPARED_CONFIG.json',V4/'MAIN_AGENT_APPROVAL_GPU7.json',ids_path]
    return dict(scope='GPU6_ONLY_SHARDS_2_4',included_gpu=6,included_shards=[2,4],
        separately_owned_gpu=7,separately_owned_shards=[3,5],separate_runtime='runtime_v4',
        separate_shards_certified_by_this_merge=False,original_gpu7_failure_preserved=True,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},old_receipt_sha256=handoff['old_receipt_sha256'])


def validate_gpu6_closed(result):
    assert result['error'] is None,'GPU6_NOT_SUCCESSFULLY_CLOSED'
    assert result['restoration'].get('restored') is True,'GPU6_HOLDER_NOT_RESTORED'
    assert len(result['workers'])==2
    assert {w['shard'] for w in result['workers']}=={2,4}
    assert all(w['returncode']==0 for w in result['workers']),'GPU6_WORKER_FAILURE'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def build():
    """Import original CPU modules in an isolated temporary module namespace."""
    saved={k:sys.modules.get(k) for k in ('transport','common','safe_size')}
    paths=list(sys.path)
    try:
        sys.path.insert(0,str(V3))
        transport=load('transport',V3/'transport.py');sys.modules['transport']=transport
        original=load('gpu6_merge_original_common',V3/'common.py')
        assert original.LANES=={6:(2,4),7:(3,5)} and original.SELECTED_SHARDS==(2,3,4,5)
        proxy=types.SimpleNamespace(**original.__dict__)
        proxy.LANES={6:(2,4)};proxy.SELECTED_SHARDS=(2,4)
        sys.modules['common']=proxy
        sizing=load('safe_size',V3/'safe_size.py');sys.modules['safe_size']=sizing
        source=transport.merge_source()
        source=transport.exact(source,"out=HERE/'merge'","out=ADAPTER/'run_v1'")
        source=transport.exact(source,"scope='SELECTED_WAVE_NOT_FULL_POOL'","scope='GPU6_ONLY_SHARDS_2_4_V4_OWNS_3_5'")
        source=transport.exact(source,"    c.save(out/'INPUT_HASHES.json',input_hashes)",
            "    c.save(out/'OWNERSHIP_EVIDENCE.json',OWNERSHIP)\n    input_hashes.update(OWNERSHIP['source_hashes'])\n    input_hashes.update(OWNERSHIP['old_receipt_sha256'])\n    c.save(out/'INPUT_HASHES.json',input_hashes)")
        assert source.count(".open('a')")==2
        source=source.replace(".open('a')",".open('rb')")
        module=types.ModuleType('gpu6_only_strict_merge')
        module.__file__=str(V3/'merge.py');module.ADAPTER=HERE
        exec(compile(source,module.__file__,'exec'),module.__dict__)
        assert original.LANES=={6:(2,4),7:(3,5)} and module.c is proxy
        return module,source
    finally:
        sys.path[:]=paths
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def execute():
    for path,h in read(HERE/'INPUT_LOCK.json').items():assert sha(ROOT/path)==h,'MERGE_ADAPTER_LOCK_CHANGED'
    evidence=ownership_evidence()
    lane=V3/'lanes/gpu_6'
    attempts=sorted(lane.glob('attempt_*'));assert len(attempts)==1
    result=read(attempts[0]/'RESULT.json');validate_gpu6_closed(result)
    assert result['restoration']==read(attempts[0]/'RESTORATION.json')
    module,_=build();module.OWNERSHIP=evidence
    module.execute()


if __name__=='__main__':execute()
