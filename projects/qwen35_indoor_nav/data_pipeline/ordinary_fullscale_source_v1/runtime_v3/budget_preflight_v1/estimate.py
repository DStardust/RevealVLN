"""CPU forecast from closed real per-house/source processing costs; no active edits."""
import collections
import hashlib
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
FULL=HERE.parents[1]
LINE=FULL.parents[1]
ROOT=LINE.parents[1]
OLD=FULL.parent/'ordinary_parallel_v1'


def sha(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length(job):
    path=job['episode']['reference_path']
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


def main():
    hashes={}
    def read(path):
        hashes[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    jobs={j['job_id']:j for j in read(OLD/'JOBS.json')}
    cal=collections.defaultdict(lambda:dict(seconds=0.,meters=0.,routes=0))
    wall=0.;route_seconds=0.
    for shard in range(4):
        root=OLD/'production'/f'shard_{shard:04d}'
        assert read(root/'GENERATION_COMPLETE.json')['all_jobs_terminal']
        wall+=read(root/'PROGRESS.json')['worker_elapsed_seconds']
        ledger=root/'LEDGER.jsonl';hashes[str(ledger.relative_to(ROOT))]=sha(ledger)
        raw=ledger.read_bytes();assert raw.endswith(b'\n')
        for line in raw.splitlines():
            r=json.loads(line);j=jobs[r['job_id']];v=cal[(j['scene_id'],j['source'])]
            v['seconds']+=r['elapsed_seconds'];v['meters']+=length(j);v['routes']+=1
            route_seconds+=r['elapsed_seconds']
    overhead=(wall-route_seconds)/1000;assert overhead>=0
    forecast=[]
    for shard in read(FULL/'PLAN.json')['shards']:
        rows=read(FULL/shard['jobs_path']);houses=collections.defaultdict(lambda:dict(routes=0,route_seconds=0.,source_routes=collections.Counter()))
        for j in rows:
            v=cal[(j['scene_id'],j['source'])];assert v['routes']>0 and v['meters']>0
            pred=v['seconds']/v['meters']*length(j)
            h=houses[j['scene_id']];h['routes']+=1;h['route_seconds']+=pred;h['source_routes'][j['source']]+=1
        point=sum(v['route_seconds'] for v in houses.values())+overhead*len(rows)
        forecast.append(dict(shard=shard['shard_id'],routes=len(rows),point_seconds=point,
            throughput_25percent_slower_seconds=point*1.25,old_3480_worker_limit_risk=point>3480,
            houses=houses))
    result=dict(status='PROSPECTIVE_COST_ESTIMATE_NOT_RUNTIME_APPROVAL',
        source_closed_wall_seconds=wall,source_closed_route_seconds=route_seconds,
        extra_asset_audit_io_seconds_per_route=overhead,forecast=forecast,
        per_house_source=[dict(house=h,source=s,**v) for (h,s),v in sorted(cal.items())],
        warning='The 25% scenario is not a confidence interval or guarantee. Timing scales with actual scene and reference length; no candidate is removed for slow generation.',
        active_gpu3_4_budgets_modified=False,new_limits_authorized=False,generated_records=0)
    for name,value in [('FORECAST.json',result),('INPUT_HASHES.json',hashes)]:
        with (HERE/name).open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps([dict(shard=r['shard'],point_seconds=r['point_seconds'],slow25_seconds=r['throughput_25percent_slower_seconds']) for r in forecast],indent=2))


if __name__=='__main__':main()
