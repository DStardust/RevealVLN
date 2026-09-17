"""Small real-file CPU Journal benchmark; no simulator, GPU or async writing."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('clock_benchmark_ledger',HERE/'ledger.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
path=m.RUNTIME/'runtime_journal.py'
seal={r.split(None,1)[1]:r.split(None,1)[0] for r in (m.RUNTIME/'SHA256SUMS').read_text().splitlines()}
assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['runtime_journal.py']
spec=importlib.util.spec_from_file_location('clock_benchmark_real_journal',path)
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

class Clock:
    def __init__(self):self.tick=0
    def __call__(self):return self.tick/1000.
    def advance(self):self.tick+=1

def trial(cls,out,actions,post_checks=1):
    out.mkdir();clock=Clock();fsyncs=[];proofs=[]
    limits=dict(total_actions=actions+1,total_seconds=10000,discovery_actions=actions+1,
                discovery_seconds=10000,certification_actions=actions+1,certification_seconds=10000)
    real_fsync=os.fsync
    def real_sync(fd):
        result=real_fsync(fd);fsyncs.append(fd);return result
    started=time.perf_counter()
    with patch.object(j.os,'fsync',side_effect=real_sync):
        with j.Journal(out/'journal',{'cpu_benchmark_only':True,'actions':actions}) as journal:
            ledger=cls(limits,clock=clock,persist=lambda s:journal.append('budget',s))
            clock.advance();ledger.start_bundle('cpu','certification')
            for i in range(actions):
                for _ in range(2):clock.advance();ledger.check_time()
                clock.advance();before=len(fsyncs);ledger.reserve_action()
                # The mock physical action is never invoked until actual event,
                # HEAD and directory fsync have returned and HEAD agrees on disk.
                assert len(fsyncs)-before==3,'RESERVE_REQUIRES_THREE_REAL_FSYNC'
                head=json.loads((journal.root/'HEAD.json').read_text())
                assert head['last_hash']==journal._records[-1]['hash']
                assert journal._records[-1]['payload']==ledger.snapshot()
                proofs.append({'action':i,'real_fsyncs_before_mock_action':3,'head_record_hash':head['last_hash']})
                journal.append('action_completed',{'bundle':'cpu','mock_action':i})
                for _ in range(post_checks):clock.advance();ledger.check_time()
            clock.advance();ledger.finish_phase();final=ledger.snapshot()
            records=list(journal._records)
    elapsed=time.perf_counter()-started
    head=json.loads((out/'journal/HEAD.json').read_text());raw=(out/'journal/events.jsonl').read_bytes()
    assert head['count']==len(records) and head['byte_length']==len(raw)
    kinds=Counter(r['kind'] for r in records)
    result={'implementation':cls.__name__,'actions':actions,'successful_final_snapshot':final,
        'budget_records':kinds['budget'],'all_journal_records':len(records),'journal_bytes':len(raw),
        'real_fsync_calls_acknowledged':len(fsyncs),'wall_seconds':elapsed,
        'every_reservation_three_actual_fsync_before_mock_action':True,'reserve_proofs':proofs,
        'journal_head_sha256':hashlib.sha256((out/'journal/HEAD.json').read_bytes()).hexdigest(),
        'journal_events_sha256':hashlib.sha256(raw).hexdigest()}
    save(out/'RESULT.json',result);return result

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def benchmark(out,actions=48,repeats=3,post_checks=1):
    out=Path(out).absolute();assert out==out.resolve() and out.is_relative_to(HERE) and out!=HERE
    assert 1<=actions<=128 and 1<=repeats<=4 and 0<=post_checks<=3
    out.mkdir();pairs=[]
    for i in range(repeats):
        order=[('baseline',m.SingleCommitBudgetLedger),('batched',m.ClockBatchingBudgetLedger)]
        if i%2:order.reverse()
        result={name:trial(cls,out/(str(i)+'_'+name),actions,post_checks) for name,cls in order}
        a,b=result['baseline'],result['batched']
        assert a['successful_final_snapshot']==b['successful_final_snapshot']
        pairs.append({'order':[n for n,_ in order],'baseline':a,'batched':b,
                      'same_successful_snapshot':True,'wall_ratio_baseline_over_batched':a['wall_seconds']/b['wall_seconds']})
    report={'status':'CPU_JOURNAL_BENCHMARK_COMPLETE_NOT_RUNTIME_ADMISSION','actions_per_trial':actions,'paired_repeats':repeats,
        'loop':'two successful time checks, one durable reserve, one synchronous mock action event, '+str(post_checks)+' time checks',
        'clock':'deterministic CPU clock for exact snapshot comparison; wall measured independently with perf_counter',
        'baseline_median_wall_seconds':statistics.median(p['baseline']['wall_seconds'] for p in pairs),
        'batched_median_wall_seconds':statistics.median(p['batched']['wall_seconds'] for p in pairs),
        'median_paired_wall_ratio':statistics.median(p['wall_ratio_baseline_over_batched'] for p in pairs),
        'baseline_budget_records_per_trial':pairs[0]['baseline']['budget_records'],
        'batched_budget_records_per_trial':pairs[0]['batched']['budget_records'],
        'same_successful_snapshots_all':True,'all_reserves_actual_three_fsync_before_mock_action':True,
        'gpu_used':False,'physical_throughput_speedup_claim':False,'runtime_integrated':False,'pairs':pairs}
    save(out/'REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2));return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--actions',type=int,default=48);p.add_argument('--repeats',type=int,default=3);p.add_argument('--post-checks',type=int,default=1)
    a=p.parse_args();benchmark(a.output,a.actions,a.repeats,a.post_checks)
