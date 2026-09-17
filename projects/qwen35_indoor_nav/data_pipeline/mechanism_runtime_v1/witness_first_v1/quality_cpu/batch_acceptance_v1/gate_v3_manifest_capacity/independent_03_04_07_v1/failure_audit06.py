"""Read-only closed failure receipt; never reclassifies incomplete work as PASS."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('failure06_sealed_gate',HERE.parent/'gate.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
old=m.gate.old
RUN=HERE.parents[3]/'batch_execution_v1/batch_06r1/run_v1'

def main():
    out=HERE/'batch_06r1';out.mkdir(exist_ok=False)
    cfg=old.read(RUN/'EXECUTION_CONFIG.json');final=old.read(RUN/'SUPERVISOR_RESULT.json')
    assert final['returncode']==-15 and final['error']=="AssertionError('MEMORY_ACCOUNTING')"
    row=m.gate.observe_run(RUN,out);old.save(out/'BATCH_REPORT.json',row)
    raw=(RUN/'journal/events.jsonl').read_bytes();head=old.read(RUN/'journal/HEAD.json')
    records=old.parse_journal(raw[:head['byte_length']],head,cfg)
    kinds=Counter(r['kind'] for r in records)
    budget=[r['payload'] for r in records if r['kind']=='budget'][-1]
    samples=[json.loads(line) for line in (RUN/'RESOURCE_SAMPLES.jsonl').read_text().splitlines()]
    proc=old.read(RUN/'PROCESS.json')
    checks=[old.gpu_snapshot(sample,proc['pid'],cfg['gpu_uuid']) for sample in samples]
    traces=[]
    for path in sorted((RUN/'bundles').glob('*/traces/*.json')):
        trace=old.read(path);sha=old.sha(path)
        matches=[r for r in records if r['kind']=='trace_saved' and r['payload'].get('sha256')==sha]
        traces.append({'path':str(path),'sha256':sha,'complete':trace.get('complete'),
            'journal_binding_count':len(matches),'action_count':len(trace['actions']),
            'observation_count':len(trace['observations']),'not_accepted_training_data':True})
    inventory=[]
    for path in sorted(RUN.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():inventory.append({'path':str(path),'bytes':path.stat().st_size,'sha256':old.sha(path)})
    source=(HERE.parents[4]/'compact_loop_v2/run.py').read_text()
    assert source.index('snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)')<source.index("with (OUT/'RESOURCE_SAMPLES.jsonl').open('a')")
    report={'status':'CLOSED_RESOURCE_FAILURE_PARTIAL_EVIDENCE_ONLY','quality_pass':False,
        'cohort_member_must_remain_failed':True,'replacement_authorized':False,
        'source_result_missing':not (RUN/'result.json').exists(),'worker_supervisor_result':final,
        'committed_journal_chain_head_verified':True,'committed_journal_kinds':dict(kinds),
        'uncommitted_tail_bytes':len(raw)-head['byte_length'],
        'committed_budget_bundles':budget['bundles'],'last_reserved_action_count':budget['total_reserved_actions'],
        'confirmed_actions_in_committed_prefix':kinds['action_completed'],
        'zero_physical_attempt_claim':False,'trace_files':traces,
        'saved_resource_samples':samples,'saved_samples_original_gpu_checks_pass':bool(checks),
        'failed_trigger_raw_xml_available':False,'failed_trigger_parsed_sample_available':False,
        'missing_evidence':['raw XML or parsed snapshot from the check_gpu call that raised MEMORY_ACCOUNTING'],
        'reason_for_missing_sample':'Original supervisor calls check_gpu before appending sample; failure stores repr(exception), not failing snapshot.',
        'actual_memory_anomaly_root_cause':'UNDETERMINED_NO_FAILED_SAMPLE',
        'no_inference_of_real_vram_excess':True,'lease_result':old.read(RUN/'LEASE_RESULT.json'),
        'restoration':old.read(RUN/'RESTORATION.json'),'file_inventory':inventory,
        'gpu_operations':0,'scientific_pass':False,'training_admission':False}
    old.save(out/'FAILURE_AND_PARTIAL_RECEIPT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('file_inventory','trace_files','worker_supervisor_result','restoration')},indent=2))

if __name__=='__main__':main()
