import collections
import copy
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
ROOT=RUNTIME.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=HERE.parent/'compatibility_cpu/PROGRAM_CANDIDATES.json'
    raw=json.loads(source.read_text());rows=[];refs={str(source):sha(source)}
    def trace(ref):
        p=ROOT/ref['path'];assert p.resolve().is_relative_to(ROOT)
        assert sha(p)==ref['sha256'];refs[str(p)]=ref['sha256'];return json.loads(p.read_text())
    for index,row in enumerate(raw):
        if not all(row.get(k) for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant')):continue
        actions=[trace(row[k])['actions'] for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant')]
        a,b=map(collections.Counter,actions[:2])
        rank=(abs((a['L']-a['R'])-(b['L']-b['R'])),abs(a['F']-b['F']),max(map(len,actions)),index)
        rows.append((rank,index,row,actions))
    rank,index,row,actions=min(rows)
    terminal=trace(row['terminal_only_prefix'])['actions'][:row['terminal_only_prefix']['prefix_cutoff']]
    oldcfg=RUNTIME/'feedback_generation_v1/run_v1/EXECUTION_CONFIG.json'
    cfg=json.loads(oldcfg.read_text());base=next(r for r in cfg['candidates'] if r['house_id']==row['house_id'])
    inventory=RUNTIME/'feedback_generation_v1/run_v1/bundles'/base['candidate_id']/'SEMANTIC_INVENTORY.json'
    objects=json.loads(inventory.read_text())['objects']
    eligible={role:sorted(int(k) for k,obj in objects.items() if obj['mpcat40']==s['mpcat40'] and obj['room']==s['room'] and obj['raw'].lower().replace('#',' ').strip()==s['raw_match']['value']) for role,s in row['roles'].items()}
    assert all(eligible.values())
    candidate=dict(candidate_id='WF_ASSEMBLY_'+str(index).zfill(3),house_id=row['house_id'],roles=row['roles'],tasks=row['tasks'],expected_eligible=eligible,
        assets=base['assets'],scene_glb=base['scene_glb'],components=dict(a=actions[0],b=actions[1],i=actions[2],terminal=terminal),
        configuration=dict(u_position=row['initial_position'],yaw_bin=row['initial_yaw_bin'],public_tail='LRLRLRLR'),
        component_provenance=dict(source_program_index=index,ranking=list(rank),source_sha256=sha(source),interpretation='FIT-only discovery, not unseen evaluation'))
    cfg.update(node='Q35N_WITNESS_FIRST_BALANCED_ASSEMBLY_V1',gpu_device=2,gpu_uuid='GPU-be1b30d0-517b-b079-871b-de195d35a1a2',candidates=[candidate])
    cfg['budget']=dict(total_actions=30000,total_seconds=2700,discovery_actions=15000,discovery_seconds=1200,certification_actions=20000,certification_seconds=1500)
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    paths=list(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md',oldcfg,inventory,out/'EXECUTION_CONFIG.json']
    paths+=list((HERE.parent/'budget_and_trace').glob('*.py'))+[HERE.parent/'budget_and_trace/SHA256SUMS']
    paths+=list((RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu').glob('*.py'))+[RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/SHA256SUMS']
    oldlock=json.loads((RUNTIME/'compact_loop_v2/run_v1/INPUT_LOCK.json').read_text())
    for p,h in oldlock.items():assert sha(p)==h,p
    refs.update(oldlock);refs.update({str(p):sha(p) for p in paths});refs.update(base['assets'])
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump(refs,f,indent=2)
    print(json.dumps(dict(prepared=True,program_index=index,ranking=rank,house=row['house_id'],gpu=2)))
if __name__=='__main__':main()
