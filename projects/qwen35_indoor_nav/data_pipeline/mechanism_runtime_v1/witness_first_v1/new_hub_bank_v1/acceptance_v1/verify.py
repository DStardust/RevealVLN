"""Readback of real closed-source bank; candidate capacity is not family yield."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent;BANK=HERE.parent;WF=BANK.parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('newhub_bank_acceptance',BANK/'prepare.py');m=p.load()
def main():
    assert not (HERE/'result.json').exists(),'NO_OVERWRITE'
    lock=m.read(BANK/'language_ready_v1/SOURCE_LOCK.json')
    for path,h in lock.items():assert m.sha(Path(path))==h,path
    original=m.read(BANK/'snapshot_v1/CONFIG_DRAFT.json');language=m.read(BANK/'language_ready_v1/CONFIG_DRAFT.json')
    expected,revisions=m.normalize(original,m.load_language());expected['node']=language['node'];assert expected==language
    counts=collections.Counter((r['house_id'],r['hub_index']) for r in language['candidates'])
    assert len(counts)==4 and all(n<=24 for n in counts.values())
    ids={r['canonical_program_id'] for r in language['candidates']};assert len(ids)==len(language['candidates'])
    prior=[];extra={}
    for root in (WF/'multi_program_bank_v1/snapshot_v3',WF/'multi_program_bank_v2/snapshot_v1',WF/'multi_program_bank_v3/snapshot_v1'):
        file=root/'CONFIG_DRAFT.json';extra[str(file)]=m.sha(file);prior.extend(m.read(file)['candidates'])
    assert not ids & {r['canonical_program_id'] for r in prior}
    assert len({r['canonical_program_id'] for r in prior})==346
    new_positions={tuple(r['configuration']['u_position']) for r in language['candidates']}
    old_positions={tuple(r['configuration']['u_position']) for r in prior if r['house_id']==p.HOUSE}
    assert all(math.dist(a,b)>=1 for a in new_positions for b in old_positions)
    ledger=m.read(BANK/'snapshot_v1/PROPOSAL_LEDGER.json');res=m.read(BANK/'snapshot_v1/result.json')
    assert len(ledger)==sum(h['programs_enumerated'] for h in res['hubs'])
    assert all(h['programs_truncated']==0 for h in res['hubs'])
    core=load('newhub_original_quality_core',WF/'multi_program_bank_v1/core.py')
    records=[json.loads(x) for x in (p.SOURCE/'houses'/p.HOUSE/'BANK_RECORDS.jsonl').read_text().splitlines()]
    closed=0
    for record in records:
        trace=record['trace'];path=core.ROOT/record['trace_ref']
        assert lock[str(path)]==m.sha(path) and m.read(path)==trace and core.compiler.complete(trace)
        closed+=core.closed(trace['observations'][0]['pose'],trace['observations'][-1]['pose'])
    source=m.read(p.SOURCE/'result.json');assert len(records)==source['counts']['complete_motion_traces']
    assert closed==source['counts']['stored_closed_loops']
    report={'status':'ACTUAL_NEW_HUB_COMPONENTS_AND_CPU_BANK_RECHECKED','source_lock_verified':len(lock),
        'actual_new_hubs':4,'complete_motion_traces':len(records),'closed_motion_loops_including_public_tail':closed,
        'outbound_complete':source['counts']['outbound_complete'],'compact_complete_and_closed':source['counts']['compact_complete_and_closed'],
        'collisions':source['counts']['trace_collisions'],'source_actual_actions':source['counts']['actual_actions'],
        'distinct_new_semantic_program_candidates':len(ids),'all_proposals_recorded':len(ledger),
        'proposal_status_counts':dict(collections.Counter(r['status'] for r in ledger)),
        'instruction_changes':sum(len(r['changes']) for r in revisions),
        'prior_candidate_pool':346,'combined_candidate_pool_no_overlap':len(prior)+len(ids),
        'total_closed_houses_unchanged':12,'combined_scout_hubs':28,'combined_hubs_with_candidates':24,
        'new_certified_families':0,'full_family_replay_required':27,'spin_neutrality_remains_untested':True,
        'same_house_hub_variants_correlated':True,'gpu_operations':0,'scientific_pass':False}
    with (HERE/'result.json').open('x') as f:json.dump(report,f,indent=2)
    extra.update({str(BANK/'language_ready_v1/SOURCE_LOCK.json'):m.sha(BANK/'language_ready_v1/SOURCE_LOCK.json'),
        str(HERE/'verify.py'):m.sha(HERE/'verify.py'),str(HERE/'result.json'):m.sha(HERE/'result.json')})
    with (HERE/'SOURCE_LOCK.json').open('x') as f:json.dump(extra,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
