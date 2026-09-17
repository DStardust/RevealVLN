"""Read-only count classification over an actually closed, hash-checked journal."""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('clock_facts_benchmark',HERE/'benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def analyze(run,output):
    run=Path(run).absolute();output=Path(output).absolute()
    assert run==run.resolve() and run.is_relative_to(HERE.parent)
    assert output==output.resolve() and output.is_relative_to(HERE)
    cfg=json.loads((run/'EXECUTION_CONFIG.json').read_text());sup=json.loads((run/'SUPERVISOR_RESULT.json').read_text())
    assert sup['returncode']==0 and sup['error'] is None and sup['cleanup_complete'] is True
    head=json.loads((run/'journal/HEAD.json').read_text());count=0;total=0;previous='0'*64;kinds=Counter();phases=defaultdict(Counter)
    previous_budget=None;active=None;file_hash=hashlib.sha256()
    with (run/'journal/events.jsonl').open('rb') as stream:
        for line in stream:
            assert line.endswith(b'\n');file_hash.update(line);total+=len(line)
            row=json.loads(line);body={k:v for k,v in row.items() if k!='hash'}
            assert b.j._canonical(row)+b'\n'==line and row['seq']==count and row['prev']==previous
            assert b.j._hash(body)==row['hash'];previous=row['hash'];count+=1;kinds[row['kind']]+=1
            if count==1:assert row['kind']=='__config__' and row['payload']==cfg
            if row['kind']=='budget':
                value=row['payload'];active=value['active'];phase=active[1] if active else 'none'
                if previous_budget is None:kind='initial'
                elif {k:v for k,v in previous_budget.items() if k!='last_clock'}=={k:v for k,v in value.items() if k!='last_clock'}:kind='clock_only'
                elif value['total_reserved_actions']>previous_budget['total_reserved_actions']:kind='reservation'
                else:kind='phase_boundary'
                phases[phase][kind]+=1;previous_budget=value
            elif row['kind']=='action_completed':phases[active[1] if active else 'none']['confirmed_actions']+=1
    assert head['count']==count and head['byte_length']==total and head['last_hash']==previous
    assert head['config_hash']==b.j._hash(cfg)
    cert=phases['certification'];report={'source_run':str(run),'source_journal_sha256':file_hash.hexdigest(),
        'source_head_sha256':hashlib.sha256((run/'journal/HEAD.json').read_bytes()).hexdigest(),
        'canonical_full_chain_and_head_verified':True,'record_kinds':dict(kinds),'budget_by_phase':{k:dict(v) for k,v in phases.items()},
        'certification_clock_only_per_confirmed_action':cert['clock_only']/cert['confirmed_actions'],
        'certification_all_budget_per_confirmed_action':(cert['clock_only']+cert['reservation']+cert['phase_boundary'])/cert['confirmed_actions'],
        'observed_component_time_cost_measured':False,'gpu_used':False,
        'interpretation':'roughly 3 successful clock-only plus 1 reservation budget record per certified action; no per-record wall timing inferred'}
    b.save(output,report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.run,a.output)
