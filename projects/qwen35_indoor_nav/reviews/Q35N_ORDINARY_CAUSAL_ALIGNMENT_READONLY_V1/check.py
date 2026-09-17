"""Stratified FIT-only causal alignment spot-check, not a new full corpus certification."""
import hashlib,json,os,random,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
snap=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/TRAINING_INDEX.jsonl'
rows=[json.loads(x) for x in snap.read_text().splitlines()]
chosen=[]
for source in sorted({r['source'] for r in rows}):
    pool=[r for r in rows if r['source']==source];random.Random(1209).shuffle(pool);seen=set()
    for r in pool:
        key=r['physical_source_route_sha256']
        if key in seen:continue
        seen.add(key);chosen.append(r)
        if len(seen)==8:break
assert len(chosen)==24 and all(r['split']=='FIT' for r in chosen)
adapter=LINE/'sft_acceptance/ordinary_baseline_v2/data.py';a=runpy.run_path(str(adapter))
count=decoded=0;results=[]
for row in chosen:
    record=a['OrdinaryRecord'](row)
    p=json.loads((ROOT/row['sourceRoot']/row['policy_file']).read_text())
    s=json.loads((ROOT/row['sourceRoot']/row['supervision_file']).read_text())
    assert len(s['frames'])==len(s['actions'])==len(record)
    acts=[a['normalize_action'](x) for x in s['actions']]
    refs=p['rgb_sequence']
    assert [Path(x).stem for x in refs]==[x['rgb_sha256'] for x in s['frames']]
    for t in range(len(record)):
        d=record.decision_metadata(t)
        assert d['control']['decision_step']==t and d['control']['rgb_refs']==refs[max(0,t-1):t+1]
        assert d['policy']==dict(instruction=p['instruction'],executed_actions=acts[max(0,t-8):t])
        assert d['supervision']['target_action']==acts[t]
        count+=1
    for t in sorted({0,len(record)//2,len(record)-1}):
        d=record.decision(t)
        assert [hashlib.sha256(im.tobytes()).hexdigest() for im in d['policy']['images']]==[Path(x).stem for x in refs[max(0,t-1):t+1]]
        decoded+=len(d['policy']['images'])
    results.append(dict(record_id=row['record_id'],source=row['source'],house=row['scene_group'],decisions=len(record),
        policy_sha256=row['policy_sha256'],supervision_sha256=row['supervision_sha256']))
out=dict(status='PASS',unix=time.time(),selection='seed1209 eight distinct physical routes per source, FIT only',
    routes=24,sources=3,decisions_checked=count,rgb_decodes=decoded,causal_projection='RGB through t, executed actions strictly before t, target action t',
    no_current_target_in_executed_history=True,not_full_corpus_recertification=True,training_updates=0,simulator_actions=0,
    index_sha256=sha(snap),adapter_sha256=sha(adapter),code_sha256=sha(Path(__file__)),records=results)
with (HERE/'RESULT.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False))
