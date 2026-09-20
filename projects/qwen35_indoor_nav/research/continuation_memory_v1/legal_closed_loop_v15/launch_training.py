"""Run only the frozen finite feature screen, under the existing owned-process guard."""
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
resource=c.load('v12_screen_resources',V5/'launch.py')


def main():
    config=c.read(HERE/'TRAIN_PROTOCOL.json')
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    assert c.sha(HERE/'DATA.json')==config['data_sha256']
    assert c.read(HERE/'features_run_001/FEATURE_RESULT.json')['parameters_unchanged']
    for relative,digest in config['source_hashes'].items():
        assert c.sha(LINE/relative)==digest, relative
    lock=(V5/'gpu_inference.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=resource.snapshot()[config['gpu']]
    assert gpu['uuid']==config['gpu_uuid'], 'DEVICE_BINDING_CHANGED'
    if gpu['free_mib']<4096:
        c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_CURRENT_FREE_MEMORY',
            selected=gpu,required_mib=4096,exclusive_reservation_claimed=False))
        return
    run=HERE/f'train_run_{len(list(HERE.glob("train_run_*")))+1:03d}'
    run.mkdir();(run/'source').mkdir()
    for path in [*HERE.glob('*.py'),HERE/'TRAIN_PROTOCOL.json',HERE/'DATA_SUMMARY.json']:
        shutil.copy2(path,run/'source'/path.name)
    base=c.read(V5/'PROTOCOL.json')
    c.write(run/'CONFIG.json',dict(base,gpu=config['gpu'],gpu_uuid=gpu['uuid'],
                                  source_hashes=config['source_hashes']),True)
    c.write(run/'PREFLIGHT.json',dict(selected=gpu,shared_allowed=True,
        expected_new_forwards=0,optimizer_updates=config['optimizer_updates'],
        hour_limit=None,concurrent_project_gpu_experiments=False,
        authorization='User requests method-benefit work without GPU-hour constraint; existing input/model frozen.'),True)
    began=time.monotonic();proc=None;identity=None;reason=None
    with (run/'worker.log').open('x') as log:
        try:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',
                str(HERE/'train.py'),str(run)],cwd=c.ROOT,
                env=resource.environment(run,gpu),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            identity=resource.owned.identity(proc.pid)
            c.write(run/'PROCESS.json',identity,True)
            while proc.poll() is None:
                selected=resource.snapshot()[config['gpu']]
                assert selected['uuid']==gpu['uuid']
                members=resource.owned.group_members(proc.pid);pids={x['pid'] for x in members}
                own=sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                foreign=[x for x in selected['processes'] if x['pid'] not in pids]
                rss=sum(x['rss_bytes'] for x in members)
                size,_=resource.owned.tree.tree_size(HERE)
                progress=c.read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {}
                if selected['free_mib']<2048:reason='GPU_MEMORY_PRESSURE'
                elif own>25*1024:reason='OWN_GPU_MEMORY'
                elif rss>64*1024**3:reason='RSS_LIMIT'
                elif size>20*1024**3:reason='OUTPUT_LIMIT'
                elif progress and time.time()-progress['unix']>900:reason='NO_PROGRESS'
                c.append(run/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=time.monotonic()-began,
                    selected=selected,own_memory_mib=own,foreign=foreign,rss_bytes=rss,
                    output_bytes=size,stop_reason=reason))
                if reason:break
                time.sleep(10)
        finally:
            cleanup=resource.owned.terminate_owned(proc,identity) if proc and identity else None
            c.write(run/'LAUNCH_RESULT.json',dict(
                status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason,returncode=proc.returncode if proc else None,
                gpu_seconds=time.monotonic()-began,cleanup=cleanup,foreign_processes_signaled=[]),True)
    print(c.read(run/'LAUNCH_RESULT.json'),flush=True)


if __name__=='__main__':
    main()
