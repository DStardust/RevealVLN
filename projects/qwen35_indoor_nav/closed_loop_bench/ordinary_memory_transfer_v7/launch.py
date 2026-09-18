"""Resume only whole native/B2/Ours triplets; no score-dependent retry or GPU-hour cap."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
V5=HERE.parent/'ordinary_cycle_pair_recovery_v5'
MEMORY=LINE/'research/continuation_memory_v1/multifamily_v7'
sys.path.insert(0,str(V5))
import common as c
resource=c.load('v7_navigation_resources',V5/'launch.py')
review=c.load('v7_navigation_review',HERE/'review.py')


def main():
    protocol=c.read(HERE/'PROTOCOL.json')
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    result=c.read(MEMORY/'run_001/RESULT.json')
    assert result['status']=='MULTIFAMILY_MATCHED_COMPARISON_COMPLETE'
    for relative,digest in protocol['source_hashes'].items():assert c.sha(LINE/relative)==digest,'SOURCE_CHANGED:'+relative
    pending=sorted(set(range(100))-set(review.committed()))
    if not pending:return
    lock=(V5/'gpu_inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=resource.snapshot()[protocol['gpu']]
    if gpu['free_mib']<12288:
        c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_FREE_MEMORY',selected=gpu,required_mib=12288))
        return
    sessions=HERE/'sessions';sessions.mkdir(exist_ok=True)
    session=sessions/f'session_{len(list(sessions.glob("session_*")))+1:03d}'
    session.mkdir();(session/'source').mkdir();(session/'frames').mkdir();(session/'pairs').mkdir()
    for path in [*HERE.glob('*.py'),HERE/'PROTOCOL.json']:shutil.copy2(path,session/'source'/path.name)
    p=c.read(V5/'PROTOCOL.json')
    memory_hashes={name:c.sha(MEMORY/'run_001'/f'{name}_1209_MEMORY.pt') for name in ('B2','Ours')}
    c.write(session/'CONFIG.json',dict(p,gpu=protocol['gpu'],gpu_uuid=gpu['uuid'],scheduled_ranks=pending,
        memory_checkpoint_sha256=memory_hashes,source_hashes=protocol['source_hashes']),True)
    c.write(session/'PREFLIGHT.json',dict(selected=gpu,scheduled_triplets=len(pending),shared_allowed=True,
        gpu_hour_limit=None,method_seed=1209,seed_selected_by_score=False),True)
    began=time.monotonic();proc=None;identity=None;reason=None
    with (session/'worker.log').open('x') as log:
        try:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'evaluate.py'),str(session)],
                cwd=c.ROOT,env=resource.environment(session,gpu),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            identity=resource.owned.identity(proc.pid);c.write(session/'PROCESS.json',identity,True)
            while proc.poll() is None:
                selected=resource.snapshot()[protocol['gpu']]
                assert selected['uuid']==gpu['uuid']
                members=resource.owned.group_members(proc.pid);pids={x['pid'] for x in members}
                own=sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                foreign=[x for x in selected['processes'] if x['pid'] not in pids]
                rss=sum(x['rss_bytes'] for x in members)
                size,_=resource.owned.tree.tree_size(HERE)
                progress=c.read(session/'PROGRESS.json') if (session/'PROGRESS.json').exists() else {}
                if selected['free_mib']<2048:reason='GPU_MEMORY_PRESSURE'
                elif own>25*1024:reason='OWN_GPU_MEMORY'
                elif rss>64*1024**3:reason='RSS_LIMIT'
                elif size>20*1024**3:reason='OUTPUT_LIMIT'
                elif progress and time.time()-progress['unix']>900:reason='NO_PROGRESS'
                c.append(session/'RESOURCE.jsonl',dict(unix=time.time(),elapsed_seconds=time.monotonic()-began,
                    selected=selected,own_memory_mib=own,foreign=foreign,rss_bytes=rss,output_bytes=size,stop_reason=reason))
                if reason:break
                time.sleep(10)
        finally:
            cleanup=resource.owned.terminate_owned(proc,identity) if proc and identity else None
            c.write(session/'LAUNCH_RESULT.json',dict(status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason,returncode=proc.returncode if proc else None,gpu_seconds=time.monotonic()-began,
                cleanup=cleanup,foreign_processes_signaled=[]),True)
    print(c.read(session/'LAUNCH_RESULT.json'),flush=True)


if __name__=='__main__':main()
