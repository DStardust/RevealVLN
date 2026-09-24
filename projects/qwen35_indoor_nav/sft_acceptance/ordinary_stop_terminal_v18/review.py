"""Split-specific full-denominator results; incomplete groups never become SR."""
import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
_cache={}
def completed_collect(stage):
    found={}
    for p in stage.glob('sessions/*/pairs/*/COLLECT.json'):
        if p not in _cache:_cache[p]=u.read(p)
        r=_cache[p];assert r['rank'] not in found and r['base_unchanged'],'INVALID_COLLECTION_SEAL';found[r['rank']]=r
    return found
def summarize(stage,verify=False):
    config=u.read(stage/'STAGE.json');n=config['planned_pairs'];found={};order=u.read(Path(config['order_path']))
    for p in sorted(stage.glob('sessions/*/pairs/*/PAIR.json')):
        if verify or p not in _cache:_cache[p]=u.read(p)
        r=_cache[p];assert r['rank'] not in found and 0<=r['rank']<n,'PAIR_REGISTRY'
        assert r['episode_id']==order[r['rank']]['episode_id'] and r['state_unchanged'] and r['input_prefix_matched'],'INVALID_PAIR_SEAL'
        if verify:
            for f,h in r['trace_hashes'].items():assert u.sha(p.parent/f)==h,'TRACE_CHANGED'
            for arm in ('A','B'):u.audit_episode(p.parent/arm,order[r['rank']]['index'],config['gt_path'])
        found[r['rank']]=r
    rows=list(found.values());complete=len(rows);arms={}
    for arm in ('A','B'):
        rr=[r[arm] for r in rows];passed=sum(e['success'] for e in rr)
        arms[arm]=dict(successes=passed,complete=complete,planned=n,sr=passed/n if complete==n else None,partial_sr=passed/complete if complete else None,identification_bounds=[passed/n,(passed+n-complete)/n],spl=sum(e['spl'] for e in rr)/complete if complete else None,ndtw=sum(e['ndtw'] for e in rr)/complete if complete else None,osr=sum(e['oracle_success'] for e in rr)/complete if complete else None,decisions=sum(e['steps'] for e in rr),collisions=sum(e['collisions'] for e in rr),failures=dict(collections.Counter(e['failure_category'] for e in rr)))
    wins=[r['episode_id'] for r in rows if r['B']['success']>r['A']['success']];losses=[r['episode_id'] for r in rows if r['B']['success']<r['A']['success']]
    houses=[]
    for h in sorted({r['house'] for r in rows}):
        subset=[r for r in rows if r['house']==h];a=sum(r['A']['success'] for r in subset);b=sum(r['B']['success'] for r in subset);houses.append(dict(house=h,complete=len(subset),A=a,B=b,delta_sr=(b-a)/len(subset)))
    return dict(status='VALID_COMPLETE' if complete==n else 'PARTIAL',phase=config['phase'],complete_pairs=complete,planned_pairs=n,missing_ranks=sorted(set(range(n))-set(found)),arms=arms,wins=wins,losses=losses,delta_sr=(len(wins)-len(losses))/n if complete==n else None,houses=houses,max_base_logit_delta=max((r['max_base_logit_delta'] for r in rows),default=None),logits_bitwise_equal=all(r['logits_bitwise_equal'] for r in rows) if rows else None,reference='V13_STOP_ONLY',candidate='V18_TRAJECTORY_STOP',adoption='NOT_AUTOMATICALLY_ADOPTED')
def main(stage):
    r=summarize(stage,True);u.write(stage/'REVIEW.json',r);return r
if __name__=='__main__':main(Path(sys.argv[1]))
