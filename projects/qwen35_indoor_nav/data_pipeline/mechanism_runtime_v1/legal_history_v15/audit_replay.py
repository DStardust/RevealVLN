"""Recompute evidence, labels and pair relations from actual raw trace files."""
import collections
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RESEARCH, c, load, raw_window,exact_pose_equal
from task_controls import terminal_only,neutral_span
import numpy as np


def main():
    run=Path(sys.argv[1]).resolve();assert run.parent==HERE
    config=c.read(run/'CONFIG.json');result=c.read(run/'RESULT.json')
    compilers=load('v15_cpu_audit_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    inspected={};families=[]
    for measured in result['families']:
        family=next(x for x in config['families'] if x['family_id']==measured['family_id'])
        compiler=compilers.Compiler(**family['compiler']);folder=run/family['family_id']
        prefixes={};queries={};rows=[];witnesses=[]
        for name,record in measured['grid'].items():
            trace=c.read(folder/(name+'.json'));h,q=name.split('__');cut=len(family['candidate']['histories'][h])
            assert trace['actions']==family['candidate']['histories'][h]+family['candidate']['continuations'][q]
            assert len(trace['actions'])<=500 and compiler.complete(trace)
            assert trace['interior_state_assignments']==0
            for o in trace['observations']:
                for kind in ('rgb','semantic'):
                    digest=o[kind+'_hash'];key=(kind,digest)
                    if key not in inspected:
                        path=run/'content'/(digest+'.'+kind+'.npy')
                        array=np.load(path,allow_pickle=False)
                        assert hashlib.sha256(array.tobytes()).hexdigest()==digest
                        assert array.shape==((224,224,3) if kind=='rgb' else (224,224))
                        assert array.dtype==(np.uint8 if kind=='rgb' else np.uint32)
                        if kind=='semantic':
                            ids,counts=np.unique(array,return_counts=True)
                            inspected[key]={str(int(i)):int(n) for i,n in zip(ids,counts)}
                        else:inspected[key]=True
                assert o['pixels']==inspected['semantic',o['semantic_hash']]
            labels={task:compiler.evaluate(trace,task) for task in compiler.tasks}
            assert labels==record['labels']
            stop=terminal_only(compiler,trace)
            if 'terminal_only' in record:assert stop==record['terminal_only']
            prefix=(raw_window(trace,cut),trace['observations'][cut]['pose'])
            assert prefixes.setdefault(h,prefix)==prefix
            query=compiler.semantic_query(compiler.query_from_trace(compiler.slice_continuation(trace,cut)))
            assert queries.setdefault(q,query)==query,'HISTORY_DEPENDENT_SUFFIX_QUERY'
            if h in family.get('neutral_span',{}):assert neutral_span(compiler,trace,family['neutral_span'][h])['pass_control']
            if h in family.get('balancing_filler_spans',{}):assert neutral_span(compiler,trace,family['balancing_filler_spans'][h])['pass_control']
            events=compiler.atoms(trace['observations'])
            if q=='C0':
                for role in ('anchor_A','anchor_B'):
                    seen=[t for t in range(cut+1) if events[t][role]]
                    if seen:
                        t=seen[0];witnesses.append(dict(history=h,role=role,first_step=t,last_step=max(seen),
                            distance_from_last_witness_to_join=cut-max(seen),instances=events[t][role],
                            rgb_sha256=[trace['observations'][t-1]['rgb_hash'],trace['observations'][t]['rgb_hash']]))
            rows.append(dict(history=h,continuation=q,decisions=len(trace['actions']),labels=labels,terminal_only=stop,
                trace_sha256=c.sha(folder/(name+'.json'))))
        if rows:
            assert all(p[0]==prefixes['H_A'][0] and exact_pose_equal(p[1],prefixes['H_A'][1]) for p in prefixes.values())
            expected=len(family['candidate']['histories'])*len(family['candidate']['continuations'])
            assert len(rows)==expected
            counts={h:dict(collections.Counter(a)) for h,a in family['candidate']['histories'].items()}
            assert counts['H_A']==counts['H_B']
            if 'H_A_N' in counts:assert counts['H_A_N']==counts['H_B_N']
            families.append(dict(family_id=family['family_id'],house=family['house'],complete_traces=len(rows),
                complete_decisions=sum(r['decisions'] for r in rows),raw_join_rechecked=True,
                suffix_queries_history_independent=True,labels=rows,visible_witnesses=witnesses,
                motion_counts=counts,neutral_controls=measured.get('neutral_controls',{}),
                relations_pass=measured.get('relations_pass'),training_admission=False))
    output=dict(status='CPU_RAW_EVIDENCE_RECOMPUTED',run=str(run.relative_to(LINE)),families=families,
        unique_arrays_rehashed=len(inspected),source_sha256=c.sha(Path(__file__)),new_model_evidence=False,
        source_protocol_sha256=c.sha(run/'CONFIG.json'),independent_test=False,training_admission=False)
    c.write(RESEARCH/(run.name+'_AUDIT.json'),output,True)
    print(dict(families=len(families),arrays=len(inspected),traces=sum(f['complete_traces'] for f in families)))


if __name__=='__main__':main()
