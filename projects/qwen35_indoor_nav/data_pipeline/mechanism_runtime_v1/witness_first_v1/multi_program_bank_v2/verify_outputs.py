"""Read-only CPU recheck of newly generated bank and finite language products."""
import collections
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('verify_bank2_prepare',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def main():
    bank=p.read(HERE/'snapshot_v1/CONFIG_DRAFT.json');language=p.read(HERE/'language_ready_v1/CONFIG_DRAFT.json')
    expected,changes=p.normalize(bank,p.load_language());expected['node']=language['node'];assert expected==language
    lock=p.read(HERE/'language_ready_v1/SOURCE_LOCK.json')
    for path,h in lock.items():assert p.sha(Path(path))==h,path
    counts=collections.Counter((r['house_id'],r['hub_index']) for r in language['candidates'])
    assert all(n<=24 for n in counts.values())
    assert len({r['canonical_program_id'] for r in language['candidates']})==len(language['candidates'])
    assert set(r['house_id'] for r in language['candidates'])==set(p.HOUSES)
    byid={r['candidate_id']:r for r in language['candidates']}
    next12=p.read(HERE/'language_ready_v1/NEXT12.json')['candidate_ids'];assert len(next12)==len(set(next12))==12
    assert all(i in byid for i in next12)
    ledger=p.read(HERE/'snapshot_v1/PROPOSAL_LEDGER.json');result=p.read(HERE/'snapshot_v1/result.json')
    assert len(ledger)==sum(h['programs_enumerated'] for h in result['hubs'])
    assert all(h['programs_truncated']==0 for h in result['hubs'])
    assert all(r['status'] is not None for r in ledger)
    assert language['runtime_allowed'] is False and language['executable'] is False and language['training_allowed'] is False
    summary={'cpu_output_recheck_pass':True,'source_lock_verified':len(lock),'candidate_count':len(byid),
        'hubs_with_candidates':len(counts),'all_proposals_recorded':len(ledger),
        'status_counts':dict(collections.Counter(r['status'] for r in ledger)),
        'instruction_changes':sum(len(r['changes']) for r in changes),'gpu_operations':0,'scientific_pass':False}
    p.save(HERE/'CPU_OUTPUT_ACCEPTANCE.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
