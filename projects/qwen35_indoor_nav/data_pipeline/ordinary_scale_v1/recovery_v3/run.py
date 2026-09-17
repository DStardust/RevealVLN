"""Bounded GPU2 resume; never signals any preexisting process."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LINE = BASE.parents[1]
ENV = LINE / '.envs/q35n_habitat_v017_g0r'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load('original_supervisor', BASE / 'run.py')
size = load('race_safe_size', HERE / 'safe_size.py')
preflight = load('recovery_preflight', HERE / 'preflight.py')

def validate_contexts(gpu, own_pid):
    external = {pid: mib for pid, mib in gpu['processes'].items() if pid != own_pid}
    assert all(0 <= mib <= 768 for mib in external.values()), 'EXTERNAL_CONTEXT_PER_PID_CAP'
    assert sum(external.values()) <= 2048, 'EXTERNAL_CONTEXT_TOTAL_CAP'
    assert gpu['memory_mib'] <= 2048 + (4096 if own_pid is not None else 0), 'TOTAL_GPU_MEMORY_CAP'
    if own_pid is not None:
        assert gpu['processes'].get(own_pid, 0) < 4096, 'GPU_CAP'
    return external

def main():
    lock_handle = (BASE / 'PRODUCER.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = preflight.verify()
    out = HERE / 'run'
    out.mkdir(exist_ok=False)
    before = old.gpu()
    old.save(out / 'GPU_BEFORE.json', before)
    assert before['utilization'] == 0, 'GPU_NOT_IDLE'
    validate_contexts(before, None)
    existing = set(before['processes'])
    initial_size = size.measure(BASE)
    assert initial_size['apparent_bytes_conservative'] < 199 * 1024 ** 3
    old.save(out / 'DISK_BEFORE.json', initial_size)
    cache = out / 'cache'
    cache.mkdir()
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CONDA_PREFIX', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONNOUSERSITE='1', XDG_CACHE_HOME=str(cache),
               NUMBA_CACHE_DIR=str(cache/'numba'), MPLCONFIGDIR=str(cache/'matplotlib'),
               TMPDIR=str(cache), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    def interrupted(signum, frame):
        raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    proc = None
    error = None
    samples = 0
    try:
        with (out/'worker.log').open('x') as log:
            proc = subprocess.Popen([str(ENV/'bin/python3'), '-I', '-B', str(HERE/'worker.py')],
                cwd=LINE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            old.save(out/'PROCESS.json', dict(pid=proc.pid, supervisor_pid=os.getpid(), started_unix=time.time()))
            while proc.poll() is None:
                time.sleep(5)
                gpu = old.gpu()
                samples += 1
                with (out/'GPU_SNAPSHOTS.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(gpu, elapsed=time.monotonic()-started))+'\n')
                external = validate_contexts(gpu, proc.pid)
                rss_result = subprocess.run(['ps','-p',str(proc.pid),'-o','rss='], capture_output=True,text=True)
                if rss_result.returncode != 0 and proc.poll() is None:
                    raise RuntimeError('RSS_MONITOR_FAILED:' + rss_result.stderr)
                rss = rss_result.stdout.strip()
                assert int(rss or 0) < 12*1024**2, 'RAM_CAP'
                elapsed = time.monotonic() - started
                assert elapsed + lock['previous_wall_seconds'] < 14400, 'TOTAL_WALL_CAP'
                sample = dict(gpu, elapsed=elapsed, external_contexts=external, nonexclusive_renderer=True)
                if samples % 12 == 0:
                    sample['disk'] = size.measure(BASE)
                    assert sample['disk']['apparent_bytes_conservative'] < 199*1024**3, 'DISK_CAP_WITH_1GIB_MARGIN'
                with (out/'RESOURCE_SAMPLES.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample)+'\n')
    except BaseException as ex:
        error = repr(ex)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        after = old.gpu()
        result = dict(returncode=proc.returncode if proc else None, error=error,
            wall_seconds=time.monotonic()-started, previous_wall_seconds=lock['previous_wall_seconds'],
            cleanup_complete=not proc or proc.pid not in after['processes'],
            holders_touched=False, holder_restoration_required=False,
            external_processes_stopped=0, gpu_after=after, scientific_pass=False)
        old.save(out/'RESULT.json', result)
        print(json.dumps(result), flush=True)
    if error or not proc or proc.returncode:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
