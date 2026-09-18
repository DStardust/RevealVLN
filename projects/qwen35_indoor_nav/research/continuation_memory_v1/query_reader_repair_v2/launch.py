"""One bounded GPU session for a fixed query-reader repair comparison."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
V5=LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.path.insert(0,str(V5))
import common as c
resource=c.load('reader_repair_resources',V5/'launch.py')


def main():
    config=c.read(HERE/'CONFIG.json')
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    lock=(V5/'gpu_inference.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=sum(c.read(p)['gpu_seconds'] for p in (HERE.parent/'pilot').glob('run_*/LAUNCH_RESULT.json'))
    prior+=sum(c.read(p)['gpu_seconds'] for p in HERE.glob('run_*/LAUNCH_RESULT.json'))
    assert prior+config['max_session_seconds']<=4*3600,'RESEARCH_BUDGET_EXHAUSTED'
    for relative,digest in config['source_hashes'].items():
        assert c.sha(LINE/relative)==digest,'SOURCE_CHANGED:'+relative
    gpu=resource.snapshot()[config['gpu']]
    if gpu['free_mib']<4096:
        c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_FREE_MEMORY',selected=gpu,required_mib=4096))
        return
    run=HERE/f'run_{len(list(HERE.glob("run_*")))+1:03d}'
    run.mkdir();(run/'source').mkdir()
    for p in list(HERE.glob('*.py'))+[HERE/'CONFIG.json']:
        shutil.copy2(p,run/'source'/p.name)
    c.write(run/'PREFLIGHT.json',dict(selected=gpu,shared_allowed=True,prior_research_gpu_seconds=prior,
        max_seconds=config['max_session_seconds'],user_authorization='Directly proceed with next repair and design after V5'),True)
    began=time.monotonic();proc=None;identity=None;reason=None
    with (run/'worker.log').open('x') as log:
        try:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'run.py'),str(run)],
                cwd=c.ROOT,env=resource.environment(run,gpu),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            identity=resource.owned.identity(proc.pid);c.write(run/'PROCESS.json',identity,True)
            while proc.poll() is None:
                selected=resource.snapshot()[config['gpu']]
                assert selected['uuid']==gpu['uuid']
                members=resource.owned.group_members(proc.pid);pids={x['pid'] for x in members}
                own=sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                foreign=[x for x in selected['processes'] if x['pid'] not in pids]
                rss=sum(x['rss_bytes'] for x in members)
                size,_=resource.owned.tree.tree_size(HERE)
                elapsed=time.monotonic()-began
                if elapsed>config['max_session_seconds']:reason='TIME_BOUND'
                elif selected['free_mib']<1024:reason='GPU_MEMORY_PRESSURE'
                elif own>3072:reason='OWN_MEMORY_BOUND'
                elif rss>16*1024**3:reason='RSS_BOUND'
                elif size>1024**3:reason='OUTPUT_BOUND'
                c.append(run/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=elapsed,selected=selected,
                    own_memory_mib=own,foreign=foreign,rss_bytes=rss,output_bytes=size,stop_reason=reason))
                if reason:break
                time.sleep(5)
        finally:
            cleanup=resource.owned.terminate_owned(proc,identity) if proc and identity else None
            c.write(run/'LAUNCH_RESULT.json',dict(status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason,returncode=proc.returncode if proc else None,gpu_seconds=time.monotonic()-began,
                cleanup=cleanup,foreign_processes_signaled=[]),True)
    print(c.read(run/'LAUNCH_RESULT.json'),flush=True)


if __name__=='__main__':
    main()
