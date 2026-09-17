"""CPU bank recheck and twelve-house candidate capacity; never certified counts."""
import collections
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('bank3_capacity_prepare',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
m=p.load();WF=HERE.parent

def hub_groups(rows):
    groups=[]
    for r in rows:
        hits=[g for g in groups if any(x['house_id']==r['house_id'] and math.dist(x['position'],r['position'])<1. for x in g)]
        merged=[r]
        for g in hits:merged.extend(g);groups.remove(g)
        groups.append(merged)
    return groups
def main():
    cfg=m.read(HERE/'snapshot_v1/CONFIG_DRAFT.json');language=m.read(HERE/'language_ready_v1/CONFIG_DRAFT.json')
    expected,changes=m.normalize(cfg,m.load_language());expected['node']=language['node'];assert expected==language
    lock=m.read(HERE/'language_ready_v1/SOURCE_LOCK.json')
    for path,h in lock.items():assert m.sha(Path(path))==h,path
    per=collections.Counter((r['house_id'],r['hub_index']) for r in language['candidates']);assert all(n<=24 for n in per.values())
    assert set(r['house_id'] for r in language['candidates'])<=set(m.HOUSES)
    assert len({r['canonical_program_id'] for r in language['candidates']})==len(language['candidates'])
    ledger=m.read(HERE/'snapshot_v1/PROPOSAL_LEDGER.json');result=m.read(HERE/'snapshot_v1/result.json')
    assert len(ledger)==sum(h['programs_enumerated'] for h in result['hubs'])
    assert all(h['programs_truncated']==0 for h in result['hubs'])
    assert all(row['status'] is not None for row in ledger)
    byid={r['candidate_id']:r for r in cfg['candidates']}
    ids=m.read(HERE/'language_ready_v1/NEXT12.json')['candidate_ids'];assert len(ids)==len(set(ids))==min(12,len(cfg['candidates']))
    assert all(i in byid for i in ids)
    assert language['runtime_allowed'] is False and language['executable'] is False
    m.save(HERE/'CPU_OUTPUT_ACCEPTANCE.json',{'cpu_recheck_pass':True,'source_lock_verified':len(lock),'candidates':len(byid),
        'candidate_hubs':len(per),'proposals':len(ledger),'status_counts':dict(collections.Counter(r['status'] for r in ledger)),
        'instruction_changes':sum(len(r['changes']) for r in changes),'gpu_operations':0,'scientific_pass':False})
    capacity_lock={};banks=[WF/'multi_program_bank_v1/snapshot_v3',WF/'multi_program_bank_v2/snapshot_v1',HERE/'snapshot_v1']
    def read(path):capacity_lock[str(path)]=m.sha(path);return m.read(path)
    all_candidates=[];hub_results=[]
    for bank in banks:
        all_candidates.extend(read(bank/'CONFIG_DRAFT.json')['candidates']);hub_results.extend(read(bank/'result.json')['hubs'])
    runs=[WF/'scout_v1/run_v1',WF/'scout_next_v1/shard_0/run_v1',WF/'bulk_source_v1/shard_00/run_v1',
        WF/'bulk_source_v1/next_shard_runtime_v1/run_v1']
    all_hubs=[];houses=[]
    for run in runs:
        closed=read(run/'result.json');assert closed['status']=='SCOUT_CLOSED' and closed['error'] is None
        for row in read(run/'EXECUTION_CONFIG.json')['candidates']:
            house=row['house_id'];houses.append(house)
            for i,hub in enumerate(read(run/'houses'/house/'FROZEN_HUB_CONFIGS.json')['hubs']):
                all_hubs.append({'house_id':house,'hub_index':i,'position':hub['position']})
    assert len(houses)==len(set(houses))==12
    assert len({r['canonical_program_id'] for r in all_candidates})==len(all_candidates)
    candidate_hubs=hub_groups([{'house_id':r['house_id'],'position':r['configuration']['u_position']} for r in all_candidates])
    counts=collections.Counter(r['house_id'] for r in all_candidates)
    inventory=[{'house_id':h,'scouted_hubs':sum(r['house_id']==h for r in all_hubs),
        'hubs_with_candidates':sum(r['house_id']==h and r['distinct_semantic_programs_selected']>0 for r in hub_results),
        'distinct_semantic_candidates':counts[h]} for h in houses]
    total=len(all_candidates);physical_hubs=len(hub_groups(all_hubs))
    summary={'scope':'CPU_CANDIDATE_INVENTORY_NOT_CERTIFIED_FAMILIES','closed_FIT_houses':12,
        'scouted_hub_records':len(all_hubs),'distinct_physical_scout_hubs_at_1m':physical_hubs,
        'physical_hubs_with_candidates_at_1m':len(candidate_hubs),'distinct_semantic_program_candidates':total,
        'per_house':inventory,'planning_target_complete_families':10000,
        'candidate_shortfall_if_every_candidate_later_passed':10000-total,
        'current_candidates_as_fraction_of_10000_not_yield':total/10000,
        'current12_house_24program_cap_upper_bound':physical_hubs*24,
        'all43_house_2hub_24program_cap_upper_bound':43*2*24,
        'all43_house_cap_shortfall_even_if_every_slot_passed':10000-43*2*24,
        'minimum_independent_hubs_for10000_at24program_per_hub_and100percent_yield':math.ceil(10000/24),
        'extra_hubs_beyond43house_2hub_plan_even_at100percent_yield':math.ceil(10000/24)-43*2,
        'current_inventory_candidate_cell_upper_bound_if_all_certified':18*total,
        'certified_families_generated_by_these_CPU_banks':0,'current_total_certified_family_count':'NOT_AUDITED_HERE',
        'same_hub_semantic_programs_are_correlated':True,'seed_or_alias_or_padding_does_not_add_family':True,
        'scouting_or_candidate_success_does_not_prove_physical_certification_yield':True,
        'planning_conclusion':'The fixed 43 houses x 2 hubs x 24 programs design cannot reach 10k unique programs. More genuinely different physical hubs and event programs would need a separately frozen source-expansion plan; existing thresholds must remain unchanged.',
        'gpu_operations':0,'scientific_pass':False}
    m.save(HERE/'TWELVE_HOUSE_CAPACITY.json',summary)
    for path in (HERE/'verify_and_capacity.py',HERE/'TWELVE_HOUSE_CAPACITY.json'):capacity_lock[str(path)]=m.sha(path)
    m.save(HERE/'CAPACITY_SOURCE_LOCK.json',capacity_lock)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
