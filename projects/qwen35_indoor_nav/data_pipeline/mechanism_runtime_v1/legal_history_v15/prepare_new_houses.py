"""Freeze houses before physical collection; no outcome-based split selection."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, ROOT, RUNTIME, RESEARCH, c, load


def main():
    planner=load('v15_semantic_metadata',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    natural=c.read(LINE/'research/continuation_memory_v1/natural_transfer_v9/DATA.json')
    pool=c.read(LINE/'research/continuation_memory_v1/semantic_transfer_v12/POOL_PROTOCOL.json')
    exposed=set(natural['fit_houses'])|set(natural['check_houses'])|set(pool['house_split'])
    exposed|={'82sE5b5pLXE','B6ByNegPMKs','PuKPg4mmafe','XcA2TqTSSAj','aayBHfsNo7d'}
    scenes=ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d'
    available=[d.name for d in scenes.iterdir() if d.is_dir() and d.name not in exposed and (d/(d.name+'.house')).exists()]
    available.sort(key=lambda h:hashlib.sha256(('Q35N_V15_LEGAL_DATA:'+h).encode()).hexdigest())
    houses=[];ineligible=[]
    for house in available:
        folder=scenes/house;objects=planner.parse_house((folder/(house+'.house')).read_text())
        groups=planner.semantic_groups(objects)
        roles={f'r{j:03d}':planner.role_spec(sig) for j,sig in enumerate(sorted(groups))}
        ids={f'r{j:03d}':[o['object_index'] for o in groups[sig]] for j,sig in enumerate(sorted(groups))}
        if len(roles)<3:
            ineligible.append(dict(house_id=house,eligible_roles=len(roles),reason='FEWER_THAN_THREE_DECLARED_ROLE_GROUPS'))
            continue
        assets={str(folder/name):c.sha(folder/name) for name in
                (house+'.glb',house+'.house',house+'.navmesh',house+'_semantic.ply')}
        houses.append(dict(house_id=house,partition='fit' if len(houses)<2 else 'check',roles=roles,
            expected_eligible=ids,scene=str(folder/(house+'.glb')),assets=assets))
        if len(houses)==4:break
    assert len(houses)==4
    config=dict(id='Q35N_V15_NEW_HOUSE_COMPONENT_COLLECTION_R1',worker='scout_new_houses.py',
        gpu=1,gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',families=[],houses=houses,
        environment_seed=0,geometry_seed=1209,output_gib=20,max_decisions=40000,
        random_hub_candidates=8,hubs_per_house=2,groups_per_hub=8,targets_per_group=1,
        max_outbound_actions=72,max_loop_actions=180,max_episode_decisions=500,
        excluded_exposed_houses=sorted(exposed),available_new_houses_order=available,
        excluded_by_semantic_metadata=ineligible,
        exposure='New to the registered Q35N memory/natural pilot house pools; base pretraining and unrelated project exposure are not claimed absent. Whole-house fit/check fixed before collection. Components are not certified families.',
        source_hashes={str(p.relative_to(LINE)):c.sha(p) for p in [*HERE.glob('*.py'),RUNTIME/'habitat_backend.py',
            RUNTIME/'guard.py',RUNTIME/'core_bridge.py',RUNTIME/'feedback_generation_v1/store.py',
            RUNTIME/'feedback_generation_v1/feedback.py',LINE/'data_pipeline/mechanism_scale_v1/planner.py',
            LINE/'data_pipeline/mechanism_factory_v2/compiler.py',LINE/'data_pipeline/mechanism_factory_v2/factory.py']})
    c.write(RESEARCH/'NEW_HOUSE_PROTOCOL.json',config,True)
    print([(h['house_id'],h['partition'],len(h['roles'])) for h in houses])


if __name__=='__main__':main()
