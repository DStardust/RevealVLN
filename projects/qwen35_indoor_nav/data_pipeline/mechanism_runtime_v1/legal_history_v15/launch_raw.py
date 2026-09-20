"""One small raw-interface run; reuse the verified shared-GPU ownership guard."""
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LINE, ROOT, RESEARCH, c
V5 = LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.modules['common'] = c
resource = c.load('v15_resources', V5/'launch.py')


def main():
    protocol_path = Path(sys.argv[1]).resolve()
    assert protocol_path.parent == RESEARCH
    config = c.read(protocol_path)
    assert all(len(h)+len(s)<=500 for row in config['families']
               for h in row['candidate']['histories'].values() for s in row['candidate']['continuations'].values())
    for path, digest in config['source_hashes'].items():
        assert c.sha(LINE/path) == digest, path
    lock = (V5/'gpu_inference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu = resource.snapshot()[config['gpu']]
    assert gpu['uuid'] == config['gpu_uuid']
    assert gpu['free_mib'] >= 4096, 'INSUFFICIENT_FREE_GPU_MEMORY'
    run = HERE/f'raw_run_{len(list(HERE.glob("raw_run_*")))+1:03d}'
    run.mkdir()
    (run/'source').mkdir()
    for path in [*HERE.glob('*.py'), protocol_path]:
        shutil.copy2(path, run/'source'/path.name)
    c.write(run/'CONFIG.json', config, True)
    c.write(run/'PREFLIGHT.json', dict(selected=gpu, shared_allowed=True,
        exclusive_reservation_claimed=False, new_model_training=False), True)
    env = resource.environment(run, gpu)
    env.pop('CUDA_VISIBLE_DEVICES', None)  # Habitat EGL takes the physical index, as in V5.
    proc = identity = None
    reason = None
    began = time.monotonic()
    with (run/'worker.log').open('x') as log:
        try:
            worker = config.get('worker', 'raw_replay.py')
            assert worker in ('raw_replay.py', 'calibrate_start.py', 'scout_new_houses.py')
            proc = subprocess.Popen([str(LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'), '-I', '-B',
                str(HERE/worker), str(run)], cwd=ROOT, env=env,
                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            identity = resource.owned.identity(proc.pid)
            c.write(run/'PROCESS.json', identity, True)
            while proc.poll() is None:
                all_gpus = resource.snapshot()
                selected = all_gpus[config['gpu']]
                assert selected['uuid'] == gpu['uuid']
                members = resource.owned.group_members(proc.pid)
                pids = {x['pid'] for x in members}
                own = sum(x['memory_mib'] or 0 for x in selected['processes'] if x['pid'] in pids)
                misplaced = [x for g in all_gpus if g['uuid'] != gpu['uuid'] for x in g['processes'] if x['pid'] in pids]
                rss = sum(x['rss_bytes'] for x in members)
                size, _ = resource.owned.tree.tree_size(HERE)
                progress = c.read(run/'PROGRESS.json') if (run/'PROGRESS.json').exists() else {}
                if misplaced: reason = 'GPU_MAPPING_ERROR'
                elif selected['free_mib'] < 2048: reason = 'GPU_MEMORY_PRESSURE'
                elif own > 25*1024 or rss > 64*2**30 or size > config['output_gib']*2**30: reason = 'RESOURCE_LIMIT'
                elif progress and time.time()-progress['unix'] > 900: reason = 'NO_PROGRESS'
                c.append(run/'RESOURCE.jsonl', dict(unix=time.time(), selected=selected, own_memory_mib=own,
                    rss_bytes=rss, output_bytes=size, misplaced=misplaced, stop_reason=reason))
                if reason: break
                time.sleep(5)
        finally:
            cleanup = resource.owned.terminate_owned(proc, identity) if proc and identity else None
            c.write(run/'LAUNCH_RESULT.json', dict(status='COMPLETE' if proc and proc.returncode==0 and not reason else 'INTERRUPTED',
                reason=reason, returncode=proc.returncode if proc else None, gpu_seconds=time.monotonic()-began,
                cleanup=cleanup, foreign_processes_signaled=[]), True)
    print(c.read(run/'LAUNCH_RESULT.json'), flush=True)
    if c.read(run/'LAUNCH_RESULT.json')['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
