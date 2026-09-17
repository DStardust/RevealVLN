"""Existing SEE2 export projected into causal feature rows and separate loss targets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
EXPORT=LINE/'data_pipeline/mechanism_runtime_v1/witness_first_v1/short_revisit_v3/run_v1/bundles/WF_SHORT_REVISIT_V3_004/export_v4'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def prepare():
    loaders=load('pilot_existing_loader',LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
    core=load('pilot_existing_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    exact=load('pilot_exact_state',HERE.parent/'check_tasks.py')
    manifest=json.loads((EXPORT/'MANIFEST.json').read_text())
    compiler=core.Compiler(**manifest['compiler_config'])
    loader=loaders.FamilyLoader(EXPORT,compiler)
    checked=loader.validate_supervision_contract()
    feature_records=[];lookup={};bindings=[]
    def feature_index(record):
        loaders.validate_policy(record)
        clean=dict(instruction=record['instruction'],rgb_refs=[r['rgb_ref'] for r in record['observations']],
                   executed=[r['action'].lower() for r in record['executed_actions']])
        key=hashlib.sha256(json.dumps(clean,sort_keys=True).encode()).hexdigest()
        if key not in lookup:
            lookup[key]=len(feature_records);feature_records.append(record);bindings.append(dict(key=key,**clean))
        return lookup[key]
    prefix_ids=list(loader.prefix_index)
    prefixes=[]
    for pid in prefix_ids:
        cell=next(x for x in loader.cells if x['prefix_id']==pid)
        record=loader.prefix_records(pid)
        trace=loader.read(cell['trace_path'])
        states=exact.exact_state_targets(compiler,trace,cell['task_id'],cell['prefix_cutoff'])
        fields=['anchor_seen_strictly_before','anchor_seen_through_current','terminal_witness_now','ordered_ready_to_stop']
        prefixes.append(dict(audit_prefix_id=pid,history_id=cell['history_id'],task_id=cell['task_id'],
            instruction=record[0]['instruction'],features=[feature_index(r) for r in record],
            state_targets=[[int(s[k]) if s[k] is not None else 0 for k in fields] for s in states],
            state_masks=[s['loss_mask'] for s in states]))
    queries=[];query_lookup={};cells=[];vocabulary=None;contexts={}
    for cell in loader.cells:
        query=compiler.encode_query(compiler.semantic_query(cell['query']))
        if vocabulary is None: vocabulary=query['vocabulary']
        assert vocabulary==query['vocabulary'],'UNBOUND_QUERY_VOCABULARY'
        token_tuple=tuple(query['token_ids'])
        if token_tuple not in query_lookup:
            query_lookup[token_tuple]=len(queries);queries.append(list(token_tuple))
        full=loader.action_stream(cell['cell_id'])
        action_by_step={t:(target,mask) for t,target,mask in zip(cell['action_steps'],cell['action_targets'],cell['action_loss_mask'])}
        tail=[]
        if any(cell['action_loss_mask']):
            for t in range(cell['prefix_cutoff'],len(full)):
                target,mask=action_by_step.get(t,('STOP',0))
                tail.append(dict(step=t,feature=feature_index(full[t]),target=['MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','STOP'].index(target),mask=mask))
        trace=loader.read(cell['trace_path']);events=compiler.atoms(trace['observations'])
        task=compiler.tasks[cell['task_id']];cut=cell['prefix_cutoff'];last=len(events)-1
        # Offline readout of future-query events only, never the historical state or Y.
        context=dict(stop_at_cutoff=last==cut,stops=trace['actions'][-1]=='S',
            suffix_anchor_before_final=any(bool(e[task['anchor']]) for e in events[cut+1:last]),
            terminal_at_final=bool(events[last][task['terminal']]) if last>cut else None)
        context_key=(cell['task_id'],cell['continuation_id'])
        assert contexts.setdefault(context_key,context)==context,'HISTORY_LEAK_IN_QUERY_CONTEXT'
        final_state=prefixes[prefix_ids.index(cell['prefix_id'])]['state_targets'][-1]
        expected=bool(context['stops'] and (final_state[3] if context['stop_at_cutoff'] else
                      context['terminal_at_final'] and (final_state[1] or context['suffix_anchor_before_final'])))
        assert not cell['bce_mask'] or int(expected)==cell['y'],'EXACT_STATE_NOT_SUFFICIENT'
        cells.append(dict(audit_cell_id=cell['cell_id'],prefix=prefix_ids.index(cell['prefix_id']),
            query=query_lookup[token_tuple],y=cell['y'] if cell['y'] is not None else 0,mask=cell['bce_mask'],
            tail=tail,outcome=cell['outcome'],query_context=context))
    assert sum(x['mask'] for cell in cells for x in cell['tail'])==492
    audit=dict(scope='DEBUG_ONLY_EXISTING_EXPOSED_SEE2_FAMILY',export=str(EXPORT.relative_to(LINE)),
        manifest_sha256=hashlib.sha256((EXPORT/'MANIFEST.json').read_bytes()).hexdigest(),
        original_training_admission=manifest['training_admission'],original_scientific_pass=manifest['scientific_pass'],
        labels_checked=checked,unique_causal_inputs=len(feature_records),prefixes=len(prefixes),cells=len(cells),
        source_semantics='SEE2, not physical room visit; old normalization and missing negative controls retained',
        independent_evaluation=False,exact_state_to_query_labels_verified=18,feature_bindings=bindings)
    return loader,feature_records,dict(prefixes=prefixes,cells=cells,queries=queries,query_vocabulary=vocabulary,audit=audit)


if __name__=='__main__':
    _,_,data=prepare()
    with (HERE/'DATA.json').open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in data['audit'].items() if k!='feature_bindings'},ensure_ascii=False))
