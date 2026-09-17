"""Compare mined role witnesses against the unchanged actual Compiler.atoms."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('witness_validation',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def main():
    snapshot=json.loads((HERE/'SNAPSHOT.json').read_text())
    summaries={row['path']:row for row in json.loads((HERE/'TRACE_WITNESSES.json').read_text())}
    compared=0
    for row in snapshot['traces']:
        trace,h=audit.stable_json(audit.ROOT/row['path']);assert h==row['sha256']
        objects=audit.stable_json(audit.ROOT/row['inventory'])[0]['objects']
        groups=audit.role_groups(objects)
        roles={json.dumps(sig,separators=(',',':')):sig[:2] for sig in groups}
        eligible={json.dumps(sig,separators=(',',':')):ids for sig,ids in groups.items()}
        role=next(iter(roles))
        compiler=audit.Compiler(roles,{'test':{'anchor':role,'terminal':role,'instruction':'CPU comparison only'}},eligible)
        first={}
        for step,events in enumerate(compiler.atoms(trace['observations'])):
            for role,ids in events.items():
                assert ids is not None
                if ids:first.setdefault(role,[step,min(ids)])
        assert first==summaries[row['path']]['role_first'],row['path']
        compared+=1
    programs=json.loads((HERE/'PROGRAM_CANDIDATES.json').read_text())
    conditions=0
    for program in programs:
        for key in ('witness_A','witness_B','terminal_only_prefix','irrelevant_prefix',
                    'closed_loop_A','closed_loop_B','closed_loop_irrelevant'):
            witness=program[key]
            if witness is None:continue
            row=summaries[witness['path']]
            assert row['start_key']==program['start_key']
            assert row['role_first'][witness['required_role']][0]<=witness['prefix_cutoff']
            assert all(role not in row['role_first'] or row['role_first'][role][0]>witness['prefix_cutoff']
                       for role in witness['forbidden_roles'])
            if witness['full_closed_loop']:assert row['closed_loop'] and row['actions']<=504
            conditions+=1
        neutral=summaries[program['neutral_public_tail']]
        for role in ('anchor_A','anchor_B','terminal'):
            spec=program['roles'][role]
            signature=json.dumps([spec['mpcat40'],spec['room'],spec['raw_match']['value']],separators=(',',':'))
            assert signature not in neutral['role_first']
    audit.save('VALIDATION.json',dict(actual_compiler_atom_parity_traces=compared,
        exported_programs_validated=len(programs),witness_conditions_checked=conditions,
        all_checks_pass=True,simulation_performed=False,physical_family_certification=False))
    print(json.dumps({'actual_compiler_atom_parity_traces':compared,'witness_conditions_checked':conditions,'all_checks_pass':True}))

if __name__=='__main__':main()
