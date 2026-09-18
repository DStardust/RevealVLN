"""Resume only whole native/B1/B2/Ours/N0 groups; no score-dependent retry or GPU-hour cap."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
V5=HERE.parent/'ordinary_cycle_pair_recovery_v5'
MEMORY=LINE/'research/continuation_memory_v1/contextual_readout_v10'
sys.path.insert(0,str(V5))
import common as c
resource=c.load('v10_navigation_resources',V5/'launch.py')
review=c.load('v10_navigation_review',HERE/'review.py')


def main(seed):
    protocol=c.read(HERE/'PROTOCOL.json')
    repair=c.read(HERE/'INFRA_REPAIR_001.json')
    assert c.sha(HERE/'PROTOCOL.json')==repair['original_protocol_sha256']
    source_hashes=dict(protocol['source_hashes'],**repair['source_overrides'])
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    assert c.read(HERE/'CPU_TEST_RESULT_INFRA_001.json')['passed']
    result=c.read(MEMORY/'run_001/RESULT.json')
    assert result['status']=='MATCHED_CONTEXTUAL_READOUT_COMPLETE'
    for relative,digest in source_hashes.items():assert c.sha(LINE/relative)==digest,'SOURCE_CHANGED:'+relative
    pending=sorted(set(range(100))-set(review.committed(seed)))
    if not pending:return
    interrupted=sum(c.read(path)['status']!='COMPLETE' for path in HERE.glob('seed_*/sessions/session_*/LAUNCH_RESULT.json'))
    assert interrupted<=3,'INFRASTRUCTURE_RETRY_LIMIT'
    lock=(V5/'gpu_inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=resource.snapshot()[protocol['gpu']]
    if gpu['free_mib']<12288:
        c.write(HERE/'RESOURCE_BLOCK.json',dict(reason='INSUFFICIENT_FREE_MEMORY',selected=gpu,required_mib=12288))
        raise RuntimeError('INSUFFICIENT_FREE_MEMORY')
    sessions=HERE/f'seed_{seed}'/'sessions';sessions.mkdir(parents=True,exist_ok=True)
    session=sessions/f'session_{len(list(sessions.glob("session_*")))+1:03d}'
    session.mkdir();(session/'source').mkdir();(session/'frames').mkdir();(session/'pairs').mkdir()
    for path in [*HERE.glob('*.py'),HERE/'PROTOCOL.json']:shutil.copy2(path,session/'source'/path.name)
    shutil.copy2(HERE/'INFRA_REPAIR_001.json',session/'source/INFRA_REPAIR_001.json')
    p=c.read(V5/'PROTOCOL.json')
    memory_hashes={name:c.sha(MEMORY/'run_001'/f'{name}_{seed}_MEMORY.pt') for name in protocol['memory_arms'].values()}
    c.write(session/'CONFIG.json',dict(p,method_seed=seed,gpu=protocol['gpu'],gpu_uuid=gpu['uuid'],scheduled_ranks=pending,
        memory_checkpoint_sha256=memory_hashes,source_hashes=source_hashes,
        infrastructure_revision='INFRA_REPAIR_001',infrastructure_revision_sha256=c.sha(HERE/'INFRA_REPAIR_001.json')),True)
    c.write(session/'PREFLIGHT.json',dict(selected=gpu,scheduled_groups=len(pending),shared_allowed=True,
        gpu_hour_limit=None,method_seed=seed,seed_selected_by_score=False),True)
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
    result=c.read(session/'LAUNCH_RESULT.json')
    print(result,flush=True)
    if result['status']!='COMPLETE':raise RuntimeError('SESSION_INTERRUPTED:'+str(session))


if __name__=='__main__':
    for seed in c.read(HERE/'PROTOCOL.json')['method_seeds']:
        main(seed)
