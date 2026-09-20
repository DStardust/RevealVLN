"""Pause the verified owned V15 evaluator after the first full 18-model group."""
import os
from pathlib import Path
import signal
import sys
import time
CODE=Path(__file__).resolve().parent
HERE=CODE.parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c
owned=c.load('v15_pause_owned',c.V3/'launch.py')


def main():
    run=HERE/'continuation_run_003';owner=c.read(run/'PROCESS.json')
    reason='Measured repeated scene construction costs about 20 seconds per rollout. Preserve a whole paired condition and resume remaining conditions with family-level simulator reuse; no score-based retry.'
    c.write(run/'PLANNED_PAUSE.json',dict(reason=reason,planned_condition=0,required_complete_rollouts=18,
        requested_unix=time.time(),only_pid=owner['pid'],future_model_or_metric_change=False),True)
    deadline=time.monotonic()+3600
    while not (run/'CONDITION_000_AUDITS.json').exists():
        if (run/'FAILURE.json').exists():raise RuntimeError('PRIOR_RUN_FAILED_BEFORE_COMPLETE_GROUP')
        if time.monotonic()>deadline:raise TimeoutError('COMPLETE_GROUP_NOT_AVAILABLE')
        time.sleep(1)
    audit=c.read(run/'CONDITION_000_AUDITS.json')
    assert len(audit)==12 and all(a['input_prefix_matched'] and a['action_prefix_matched'] for a in audit)
    current=owned.identity(owner['pid']);assert current is not None
    assert all(current[k]==owner[k] for k in ('pid','pgid','sid','start_ticks'))
    assert current['pgid']==current['sid']==current['pid']
    args=Path(f'/proc/{current["pid"]}/cmdline').read_bytes().split(b'\0')
    assert str(HERE/'evaluate_continuations.py').encode() in args and str(run).encode() in args
    c.write(run/'PAUSE_SIGNAL.json',dict(reason=reason,verified_owner=current,signal='SIGINT',
        group_audit_sha256=c.sha(run/'CONDITION_000_AUDITS.json'),sent_unix=time.time()),True)
    os.kill(current['pid'],signal.SIGINT)
    print('Requested graceful Python cleanup and final state seal of the owned evaluator.',flush=True)


if __name__=='__main__':main()
