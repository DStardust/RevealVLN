"""Evidence localization for the pre-registered audit; no metric changes."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit',HERE/'audit.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

def main():
    before=a.verify_source()
    roles=a.read(a.SRC/'TASK_ROLE_INVENTORY.json'); eligible=roles['eligible']; kinds=roles['kinds']
    sup=a.rows(a.SRC/'SUPERVISION_ONLY.jsonl')
    traces={t['trace_hash']:t for t in (a.read(p) for p in (a.SRC/'physical_traces').glob('*.json'))}
    states=[]
    for s in sup:
        tr=traces[s['full_log_ref'][7:]]; cutoff=s['prefix_cutoff_step']
        ev=a.atoms(tr['observations'][:cutoff+1],eligible,256)
        anchor=next(k for k,v in kinds.items() if v==[s['task_program']['anchor_object_category'],s['task_program']['anchor_room_category']])
        seen=any(x[anchor] for x in ev); ready=any(x[anchor] for x in ev[:-1]) and bool(ev[-1]['B'])
        derived='READY_TO_STOP' if ready else 'WAIT_TERMINAL_WITNESS' if seen else 'WAIT_ANCHOR'
        actual=s['m2_program_state'][cutoff]['state']; assert derived==actual
        states.append({'sample_id':s['sample_id'],'cutoff':cutoff,'independently_derived_state':derived,'stored_state':actual})
    tr=a.read(a.SRC/'physical_traces/1109_H_K_C0.json'); obs=tr['observations']
    events192=a.atoms(obs,eligible,192); events256=a.atoms(obs,eligible,256)
    extra=[]
    for t,e in enumerate(events192):
        for idx in e['D'] or []:
            if idx not in (events256[t]['D'] or []):
                extra.append({'step':t,'mask_id':idx,'pixel_counts':[obs[t-1]['pixels'].get(idx,0),obs[t]['pixels'].get(idx,0)],
                              'rgb_hashes':[obs[t-1]['rgb_hash'],obs[t]['rgb_hash']],
                              'in_history':t<=206})
    assert before==a.verify_source()
    a.save('SUPPLEMENT_EVIDENCE.json',{'script_sha256':a.digest(HERE/'supplement.py'),'source_hashes_unchanged':True,
             'm2_causal_recomputation':states,'192_only_tv_witness_H_K_C0':extra})
    print(extra)

if __name__=='__main__': main()
