"""CPU-only actual original-English-pool coverage; no quality claims from ledger."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PIPE=HERE.parent
FULL=PIPE/'ordinary_fullscale_source_v1'
ROOT=HERE.parents[3]
def read(p):return json.loads(p.read_text())
def roots():
    return [PIPE/'ordinary_pilot_v1',PIPE/'ordinary_scale_v1']+[
        PIPE/'ordinary_parallel_v1/production'/f'shard_{s:04d}' for s in range(4)]+[
        FULL/'production'/f'shard_{s:04d}' for s in range(7)]+[
        FULL/'rescue_production'/f'shard_{s:04d}' for s in (0,1)]+[
        FULL/version/'production'/f'shard_{s:04d}' for version in ('auto_generation_v1','auto_generation_v2') for s in (2,4)]+[
        FULL/(version+'_production')/f'shard_{s:04d}' for version in ('auto_generation_v3','auto_generation_v4') for s in (2,4)]
def inspect():
    pool={};attempts={};terminal={};partials=[];summary=[];hashes={}
    for root in roots():
        jobs=read(root/'JOBS.json');byid={j['job_id']:j for j in jobs};assert len(byid)==len(jobs)
        for j in jobs:
            k=j['physical_source_route_sha256'];assert k not in pool or pool[k]==j
            pool[k]=j
        ledger=(root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b''
        assert not ledger or ledger.endswith(b'\n')
        lines=[json.loads(x) for x in ledger.splitlines()];t={r['job_id'] for r in lines};assert len(t)==len(lines)
        d={p.name for p in (root/'routes').iterdir()} if (root/'routes').exists() else set()
        assert t<=d<=set(byid)
        for ident in d:
            key=byid[ident]['physical_source_route_sha256'];assert key not in attempts,('DUPLICATE_PHYSICAL_ATTEMPT',ident)
            attempts[key]=str(root.relative_to(ROOT))
            if ident in t:terminal[key]=next(r['status'] for r in lines if r['job_id']==ident)
            else:partials.append(dict(job_id=ident,physical_route_key=key,root=str(root.relative_to(ROOT))))
        summary.append(dict(root=str(root.relative_to(ROOT)),assigned=len(jobs),terminal=len(t),partial=len(d-t),never_attempted_here=len(jobs)-len(d)))
        for p in (root/'JOBS.json',root/'LEDGER.jsonl'):
            if p.exists():hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    missing=[pool[k] for k in sorted(set(pool)-set(attempts))]
    assert len(pool)==8742
    final=FULL/'auto_generation_v4/lanes/gpu_6/attempt_000/RESULT.json'
    result=read(final);assert result['error'] is None and result['restoration']['restored'] is True
    assert [(w['shard'],w['returncode']) for w in result['workers']]==[(2,0),(4,0)]
    return dict(pool_routes=len(pool),pool_aliases=sum(len(j['instruction_alias_episodes']) for j in pool.values()),
        attempted_routes=len(attempts),terminal_routes=len(terminal),partial_routes=len(partials),
        never_attempted_routes=len(missing),never_attempted_jobs=missing,partials=partials,root_summary=summary,
        auto_v4_worker_completion_verified=True,auto_v4_holder_restoration_verified=True,
        strict_merge_ready=(FULL/'auto_generation_v4/merge/RESULT.json').exists(),
        certified_quality_not_inferred_from_coverage=True,input_hashes=hashes,training_or_gpu_started=False)
if __name__=='__main__':
    value=inspect();out=HERE/'COVERAGE.json'
    with out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k not in ('root_summary','input_hashes','partials','never_attempted_jobs')},indent=2))
