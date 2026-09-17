"""Post-audit descriptive controls; do not retune or overwrite the frozen candidates."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('witness_optional',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def flags(program,actions):
    loops={k:program[k] for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant')}
    all_closed=all(loops.values())
    i=loops['closed_loop_irrelevant']
    ia=actions[i['path']] if i else None
    distinct=all_closed and all(ia!=actions[loops[k]['path']] for k in ('closed_loop_A','closed_loop_B'))
    rooms={program['roles'][k]['room'] for k in ('anchor_A','anchor_B','terminal')}
    return dict(all_three_actual_closed_loops=all_closed,
        I_actual_closed_loop_has_at_least_2F=ia is not None and ia.count('F')>=2,
        I_actual_closed_loop_actions_differ_from_A_and_B=distinct,
        legacy_A_B_T_three_distinct_rooms=len(rooms)==3)

def main():
    summaries=json.loads((HERE/'TRACE_WITNESSES.json').read_text())
    by_path={r['path']:r for r in summaries}
    _,programs=audit.compatibility(summaries)
    paths={p[k]['path'] for p in programs for k in ('closed_loop_A','closed_loop_B','closed_loop_irrelevant') if p[k]}
    actions={}
    for path in sorted(paths):
        trace,h=audit.stable_json(audit.ROOT/path);assert h==by_path[path]['sha256']
        actions[path]=trace['actions']
    rows=[]
    for index,p in enumerate(programs):
        f=flags(p,actions)
        neutral=by_path[p['neutral_public_tail']]
        signatures=[json.dumps([p['roles'][r]['mpcat40'],p['roles'][r]['room'],p['roles'][r]['raw_match']['value']],separators=(',',':'))
                    for r in ('anchor_A','anchor_B','terminal')]
        f['public_LRLRLRLR_has_no_A_B_T_event']=neutral['public_tail_probe'] and not any(s in neutral['role_first'] for s in signatures)
        f['strict_motion_detour_component_conditions']=all(f[k] for k in ('all_three_actual_closed_loops',
            'I_actual_closed_loop_has_at_least_2F','I_actual_closed_loop_actions_differ_from_A_and_B',
            'public_LRLRLRLR_has_no_A_B_T_event'))
        rows.append(dict(original_sorted_program_index=index,house_id=p['house_id'],start_key=p['start_key'],**f))
    counts={key:sum(row[key] for row in rows) for key in rows[0] if key not in ('original_sorted_program_index','house_id','start_key')}
    audit.save('OPTIONAL_FILTERS.json',dict(scope='descriptive_postaudit_no_candidate_reselection',
        all_prefix_compatible_programs=len(programs),counts=counts,
        strict_motion_plus_legacy_three_rooms=sum(r['strict_motion_detour_component_conditions'] and r['legacy_A_B_T_three_distinct_rooms'] for r in rows),
        rows=rows,original_PROGRAM_CANDIDATES_unchanged=True,scientific_pass=False))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
