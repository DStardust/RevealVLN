"""One shared-GPU model session after all fixed training arms have finished."""
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
resource=c.load('v15_continuation_resources',V5/'launch.py')


def main():
    protocol=c.read(HERE/'CONTINUATION_PROTOCOL.json')
    assert c.read(HERE/'CONTINUATION_CPU_TEST_RESULT.json')['passed']
    assert len(c.read(HERE/'train_run_001/RESULT.json')['runs'])==18
    for relative,digest in protocol['source_hashes'].items():assert c.sha(LINE/relative)==digest,relative
    used={condition['family_id'] for condition in protocol['conditions']};assets={}
    for family in c.read(LINE/protocol['raw_config'])['families']:
        if family['family_id'] in used:assets.update(family['assets'])
    for path,digest in assets.items():
        assert Path(path).resolve().is_relative_to(c.ROOT) and c.sha(path)==digest,path
    lock=(V5/'gpu_inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=resource.snapshot()[protocol['gpu']]
    assert selected['uuid']==protocol['gpu_uuid']
    if selected['free_mib']<18000:
        c.write(HERE/'CONTINUATION_RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_FREE_MEMORY',selected=selected))
        return
    run=HERE/f'continuation_run_{len(list(HERE.glob("continuation_run_*")))+1:03d}'
    run.mkdir();(run/'source').mkdir()
    for path in [*HERE.glob('*.py'),HERE/'CONTINUATION_PROTOCOL.json']:
        shutil.copy2(path,run/'source'/path.name)
    base=c.read(V5/'PROTOCOL.json')
    c.write(run/'CONFIG.json',dict(base,gpu=protocol['gpu'],gpu_uuid=selected['uuid'],
        source_hashes=protocol['source_hashes']),True)
    c.write(run/'PREFLIGHT.json',dict(selected=selected,shared_gpu_allowed=True,expected_rollouts=216,
        train_result_sha256=c.sha(HERE/'train_run_001/RESULT.json'),exclusive_reservation_claimed=False,
        verified_scene_assets=assets),True)
    began=time.monotonic();proc=None;owner=None;reason=None
    with (run/'worker.log').open('x') as log:
        try:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',
                str(HERE/'evaluate_continuations.py'),str(run)],cwd=c.ROOT,
                env=resource.environment(run,selected),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            owner=resource.owned.identity(proc.pid);c.write(run/'PROCESS.json',owner,True)
            while proc.poll() is None:
                gpu=resource.snapshot()[protocol['gpu']];assert gpu['uuid']==selected['uuid']
                members=resource.owned.group_members(proc.pid);pids={x['pid'] for x in members}
                own=sum(x['memory_mib'] or 0 for x in gpu['processes'] if x['pid'] in pids)
                rss=sum(x['rss_bytes'] for x in members);size,_=resource.owned.tree.tree_size(HERE)
                progress=c.read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {}
                if gpu['free_mib']<2048:reason='GPU_MEMORY_PRESSURE'
                elif own>25*1024:reason='OWN_GPU_MEMORY'
                elif rss>64*1024**3:reason='RSS_LIMIT'
                elif size>20*1024**3:reason='OUTPUT_LIMIT'
                elif progress and time.time()-progress['unix']>900:reason='NO_PROGRESS'
                c.append(run/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=time.monotonic()-began,
                    selected=gpu,own_memory_mib=own,rss_bytes=rss,output_bytes=size,reason=reason))
                if reason:break
                time.sleep(10)
        finally:
            cleanup=resource.owned.terminate_owned(proc,owner) if proc and owner else None
            c.write(run/'LAUNCH_RESULT.json',dict(status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason,returncode=proc.returncode if proc else None,gpu_seconds=time.monotonic()-began,
                cleanup=cleanup,foreign_processes_signaled=[]),True)
    print(c.read(run/'LAUNCH_RESULT.json'),flush=True)
    if c.read(run/'LAUNCH_RESULT.json')['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
