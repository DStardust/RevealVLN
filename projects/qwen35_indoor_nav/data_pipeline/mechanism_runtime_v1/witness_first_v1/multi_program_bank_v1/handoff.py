"""Freeze 12 new semantic programs, excluding prior frozen programs, CPU only."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('multi_handoff_core',HERE/'core.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def read(path):return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    source=HERE/'snapshot_v3';out=HERE/'handoff_v1';assert not out.exists()
    cfg=read(source/'CONFIG_DRAFT.json');lock=read(source/'SOURCE_LOCK.json')
    for p,h in lock.items():assert c.stable(Path(p))[1]==h,'SOURCE_LOCK_CHANGED'
    known={};known_sources=[]
    paths=[c.WF/'batch_execution_v1'/name/'run_v1/EXECUTION_CONFIG.json' for name in ('batch_00','batch_01r1','batch_02r2')]
    paths.append(c.WF/'short_revisit_v3/run_v1/EXECUTION_CONFIG.json')
    for path in paths:
        raw,h=c.stable(path);lock[str(path)]=h;prior=json.loads(raw)
        for row in prior['candidates']:
            proposal={'house_id':row['house_id'],'roles':row['roles'],'hub_pose':{'position':row['configuration']['u_position']}}
            known.setdefault(c.semantic_id(proposal),[]).append(row['candidate_id'])
        known_sources.append(str(path))
    novel=[];rejections=[]
    for row in cfg['candidates']:
        if row['canonical_program_id'] in known:
            rejections.append({'candidate_id':row['candidate_id'],'reason':'SEMANTIC_PROGRAM_ALREADY_IN_PRIOR_FROZEN_CONFIG',
                'prior_candidate_ids':known[row['canonical_program_id']]})
        else:novel.append(row)
    houses=collections.defaultdict(list)
    for row in novel:houses[row['house_id']].append(row)
    for rows in houses.values():rows.sort(key=lambda r:(c.rank(r),r['hub_index'],r['canonical_program_id']))
    selected=[]
    for rank in range(max(map(len,houses.values()),default=0)):
        for house in sorted(houses):
            if rank<len(houses[house]):selected.append(houses[house][rank])
    selected=selected[:12]
    assert len({r['canonical_program_id'] for r in selected})==len(selected)
    out.mkdir()
    draft=dict(cfg,candidates=selected,node='Q35N_MULTI_PROGRAM_FIRST12_NEW_SEMANTIC_PROGRAMS')
    save(out/'CONFIG_DRAFT.json',draft)
    save(out/'SELECTION.json',{'candidate_pool':len(cfg['candidates']),'novel_semantic_pool':len(novel),
        'first_new_candidates':len(selected),'houses':sorted({r['house_id'] for r in selected}),
        'selection_rule':'exclude prior frozen category-room A/B-unordered+T semantic programs; round-robin houses then shortest representative',
        'excluded':rejections,'prior_frozen_sources':known_sources,'existing_frozen_is_not_claim_of_all_executed':True,
        'same_hub_variants_independent':False,'same_house_variants_independent':False,
        'language_realization_required_before_new_runtime_freeze':True,
        'physical_replay_required':True,'full_replays_per_family':27,'scientific_pass':False})
    for path in [source/'CONFIG_DRAFT.json',source/'SOURCE_LOCK.json',HERE/'handoff.py',out/'CONFIG_DRAFT.json',out/'SELECTION.json']:
        _,h=c.stable(path);lock[str(path)]=h
    save(out/'SOURCE_LOCK.json',lock)
    print(json.dumps({'first_new_candidates':len(selected),'novel_semantic_pool':len(novel),
        'houses':sorted({r['house_id'] for r in selected})},indent=2))
if __name__=='__main__':main()
