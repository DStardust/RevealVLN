"""A separate bounded GPU budget for the debug memory prototype."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
V5=LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.path.insert(0,str(V5))
import common as c
resource=c.load('pilot_shared_resource',V5/'launch.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1);args=parser.parse_args()
    config=c.read(HERE/'CONFIG.json')
    assert len(c.committed_pairs())>=5,'ENGINEERING_5_PAIR_CHAIN_NOT_YET_PASSED'
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    lock=(V5/'gpu_inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=sum(c.read(p)['gpu_seconds'] for p in HERE.glob('run_*/LAUNCH_RESULT.json'))
    remaining=config['max_gpu_seconds']-prior
    if remaining<180: raise RuntimeError('PILOT_BUDGET_EXHAUSTED')
    gpu=resource.snapshot()[args.gpu]
    p=c.read(V5/'PROTOCOL.json')
    original_model=str((c.TRAIN/'model.py').resolve())
    assert c.sha(original_model)==c.read(V5/'SOURCE_LOCK.json')['existing_sources'][original_model]
    # The engineering probe measured about5.2GiB for the model and <7GiB with both
    # simulators. This no-simulator pilot reserves9GiB, including >2GiB headroom.
    required_free_mib=9216
    if gpu['free_mib']<required_free_mib:
        c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_FREE_MEMORY',gpu=gpu,required_free_mib=required_free_mib,exclusive_lease_required=False))
        return
    run=HERE/f'run_{len(list(HERE.glob("run_*")))+1:03d}';run.mkdir()
    (run/'source').mkdir()
    for path in HERE.glob('*.py'):shutil.copy2(path,run/'source'/path.name)
    sources={path.name:c.sha(path) for path in HERE.glob('*.py')}
    sources.update({str(path.relative_to(LINE)):c.sha(path) for path in [V5/'evaluate.py',V5/'common.py',c.TRAIN/'model.py']})
    c.write(run/'CONFIG.json',dict(p,gpu=args.gpu,gpu_uuid=gpu['uuid'],source_hashes=sources,pilot_config=config),True)
    c.write(run/'PREFLIGHT.json',dict(selected=gpu,remaining_gpu_seconds=remaining,phase='DEBUG_MEMORY_PILOT',shared_allowed=True,required_free_mib=required_free_mib),True)
    began=time.monotonic();proc=None;initial=None;reason=None
    with (run/'worker.log').open('x') as log:
        try:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'run.py'),str(run)],
                env=resource.environment(run,gpu),cwd=c.ROOT,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            initial=resource.owned.identity(proc.pid);c.write(run/'PROCESS.json',initial,True)
            while proc.poll() is None:
                members=resource.owned.group_members(proc.pid);pids={x['pid'] for x in members}
                selected=resource.snapshot()[args.gpu]
                foreign=[x for x in selected['processes'] if x['pid'] not in pids]
                own=sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                rss=sum(x['rss_bytes'] for x in members)
                size,misses=resource.owned.tree.tree_size(HERE)
                engineering_size,_=resource.owned.tree.tree_size(V5)
                size+=engineering_size
                elapsed=time.monotonic()-began
                progress=c.read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {}
                if selected['free_mib']<p['minimum_free_mib']:reason='MEMORY_PRESSURE'
                elif elapsed>min(remaining,config['max_continuous_seconds'])-30:reason='TIME_BUDGET'
                elif rss>p['cpu_memory_gib']*1024**3:reason='RSS_BUDGET'
                elif size>config['max_output_gib']*1024**3:reason='OUTPUT_BUDGET'
                elif progress and time.time()-progress['unix']>600:reason='NO_PROGRESS'
                c.append(run/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=elapsed,gpu=selected,
                    foreign=foreign,own_memory_mib=own,rss_bytes=rss,output_bytes=size,stop_reason=reason))
                if reason: break
                time.sleep(5)
        finally:
            cleanup=resource.owned.terminate_owned(proc,initial) if proc and initial else None
            c.write(run/'LAUNCH_RESULT.json',dict(status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason,returncode=proc.returncode if proc else None,gpu_seconds=time.monotonic()-began,
                cleanup=cleanup,foreign_processes_signaled=[]),True)
    lock.close()
    print(c.read(run/'LAUNCH_RESULT.json'),flush=True)


if __name__=='__main__':main()
