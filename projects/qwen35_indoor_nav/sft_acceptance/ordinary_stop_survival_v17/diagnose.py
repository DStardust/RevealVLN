"""Read-only sealed-pair regression attribution; no feature/model replay."""
import collections,hashlib,json,math,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at(path,step):
    with path.open() as f:
        for i,line in enumerate(f,1):
            if i==step:return json.loads(line)
    raise ValueError("MISSING_STEP")
def analyze(root,tag):
    stage=root/tag;review=read(stage/'REVIEW.json');counts=collections.Counter();details=[];seals=[];steps={'A':[],'B':[]};initial_stops=collections.Counter()
    for p in sorted(stage.glob('sessions/*/pairs/*/PAIR.json')):
        r=read(p);seals.append(dict(path=str(p),sha256=sha(p)))
        counts['pairs']+=1;kind='win' if r['B']['success']>r['A']['success'] else 'loss' if r['B']['success']<r['A']['success'] else 'tie'
        counts[kind]+=1
        for a in ['A','B']:
            steps[a].append(r[a]['steps'])
            if r[a]['stopped'] and r[a]['steps']==1:initial_stops[a]+=1
        step=r['first_action_divergence']
        if step is None:counts['no_divergence']+=1;continue
        aa=at(p.parent/'A/POLICY_STEPS.jsonl',step);bb=at(p.parent/'B/POLICY_STEPS.jsonl',step)
        assert aa['raw']==bb['raw'] and aa['processed']==bb['processed'],'FIRST_INPUT_DIFF'
        assert aa['base_action']==bb['base_action'],'BASE_ARGMAX_FLIP'
        assert r['state_unchanged'] and r['input_prefix_matched']
        a,b=aa['executed_action'],bb['executed_action'];event='B_stop_A_move' if b=='STOP' else 'A_stop_B_move' if a=='STOP' else 'motion_change'
        dist=r['A']['distances'][step-1];assert dist==r['B']['distances'][step-1]
        counts[event]+=1;counts[kind+'_'+event]+=1
        if b=='STOP' and dist>=3:counts['B_first_divergence_far_STOP']+=1
        if a=='STOP' and dist>=3:counts['A_first_divergence_far_STOP']+=1
        details.append(dict(rank=r['rank'],episode_id=r['episode_id'],house=r['house'],outcome=kind,first_divergence=step,event=event,distance_before=dist,A_action=a,B_action=b,A_success=r['A']['success'],B_success=r['B']['success'],A_margin=aa['logits'][3]-max(aa['logits'][:3]),B_margin=bb['logits'][3]-max(bb['logits'][:3])))
    assert counts['pairs']==review['planned_pairs']
    return dict(counts=dict(counts),first_step_STOP=dict(initial_stops),mean_decisions={a:sum(v)/len(v) for a,v in steps.items()},metrics=review,first_divergences=details,seals=seals)
def main():
    old=LINE/'sft_acceptance/ordinary_stop_coverage_v15/runs/coverage_001';new=LINE/'sft_acceptance/ordinary_action_repair_v16/runs/motion_001';results={}
    for version,run in [('V15',old),('V16',new)]:
        for tag in ['DEV','UNSEEN']:
            results[version+'_'+tag]=analyze(run,tag)
            print(version,tag,results[version+'_'+tag]['counts'],flush=True)
    parity={}
    for tag in ['DEV','UNSEEN']:
        previous={read(p)['rank']:read(p) for p in (old/tag).glob('sessions/*/pairs/*/PAIR.json')}
        n=0
        for p in (new/tag).glob('sessions/*/pairs/*/PAIR.json'):
            r=read(p);a=previous[r['rank']]['B'];b=r['A']
            for key in ['positions','distances','success','spl','ndtw','steps','stopped','action_counts']:assert a[key]==b[key],(tag,r['rank'],key)
            n+=1
        parity[tag]=dict(pairs=n,identical_V15_policy_full_trajectories=True)
    (HERE/'DIAGNOSIS.json').write_text(json.dumps(dict(unix=time.time(),results=results,cross_run_reference_parity=parity),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
