import collections,hashlib,importlib.util,json,math,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main():
    assert not (HERE/'BUILD_RESULT.json').exists()
    k=c.load('contracts',c.DATA.parent/'contracts.py')
    assert c.read(c.DATA/'RESULT.json')['status'] in ('PASS','COMPLETE')
    episodes=c.read(c.FIT/'EPISODES_PRIVILEGED.json');assert len(episodes)==64
    original={};sources=[c.FIT/'EPISODES_PRIVILEGED.json',c.DATA/'SUPERVISION_ONLY.jsonl',c.DATA/'DATA_SEAL.json',c.DATA/'RESULT.json',c.DATA.parent/'contracts.py']
    for lane in sorted((c.FIT/'run_001/lanes').glob('lane_*')):
        files=[lane/x for x in ('POLICY_STEPS.jsonl','STEPS_PRIVILEGED.jsonl','INTERFACE.jsonl')];sources.extend(files)
        starts={x['index']:x for x in c.rows(files[2])};ps=collections.defaultdict(list);ss=collections.defaultdict(list)
        for x in c.rows(files[0]):ps[x['index']].append(x)
        for x in c.rows(files[1]):ss[x['index']].append(x)
        for idx,pp in ps.items():
            ep=lane/f'episode_{idx:02d}.json';sources.append(ep);e=c.read(ep)
            seen=[starts[idx]['rgb_sha256']];executed=[];distance=e['start_distance_m']
            assert len(pp)==len(ss[idx])==e['steps']
            for j,(p,t) in enumerate(zip(pp,ss[idx])):
                assert p['step']==t['step']==j+1 and p['action']==t['action']
                value=k.payload(episodes[idx]['instruction']['instruction_text'],seen[-2:],executed[-8:])
                original[idx,j]=dict(value=value,record_id=k.key(value),logits=p['logits'],distance=distance,house=e['house'])
                seen.append(t['rgb_sha256']);executed.append(p['action']);distance=t['distance_to_goal']
    assert len(original)==7225
    groups=collections.defaultdict(list);inputs={}
    for row in c.rows(c.DATA/'SUPERVISION_ONLY.jsonl'):
        o=original[row['source_index'],row['decision_step']];ident=row['record_id']
        assert ident==o['record_id'] and o['house']==row['scene_group']
        assert math.isfinite(row['distance_to_goal']) and abs(o['distance']-row['distance_to_goal'])<1e-5
        assert (o['distance']<3)==(row['distance_to_goal']<3)
        value=dict(record_id=ident,**o['value']);assert ident not in inputs or inputs[ident]==value;inputs[ident]=value
        groups[ident].append(dict(episode=row['source_index'],step=row['decision_step'],house=o['house'],target=int(o['distance']<3),distance=o['distance'],original_logits=o['logits']))
    assert len(groups)==5487 and sum(map(len,groups.values()))==7225
    rejected=[];labels=[];policy=[]
    for ident,occ in groups.items():
        if len({x['target'] for x in occ})!=1:rejected.append(ident);continue
        assert len({x['house'] for x in occ})==1
        assert all(x['original_logits']==occ[0]['original_logits'] for x in occ)
        policy.append(inputs[ident]);labels.append(dict(record_id=ident,target=occ[0]['target'],house=occ[0]['house'],occurrences=occ))
    houses=sorted({x['house'] for x in labels});assert len(houses)==16
    shuffled=houses.copy();random.Random(1209).shuffle(shuffled)
    assert not set(houses)&set(c.read(c.LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1/PROTOCOL.json')['houses'])
    c.jsonl(HERE/'POLICY_INPUTS.jsonl',policy);c.jsonl(HERE/'SUPERVISION_ONLY.jsonl',labels)
    c.write(HERE/'SPLIT.json',dict(seed=1209,fit_houses=shuffled[4:],heldout_houses=shuffled[:4],all_houses=houses))
    c.write(HERE/'BUILD_RESULT.json',dict(status='PASS',unix=time.time(),unique_inputs=len(policy),source_occurrences=7225,quarantined_conflicts=rejected,positive_inputs=sum(x['target'] for x in labels),houses=houses,extra_forward=0,extra_simulator_actions=0))
    c.write(HERE/'DATA_SOURCES.json',dict(files={str(p):c.sha(p) for p in sources}))
    print(json.dumps(c.read(HERE/'BUILD_RESULT.json')))
if __name__=='__main__':main()
