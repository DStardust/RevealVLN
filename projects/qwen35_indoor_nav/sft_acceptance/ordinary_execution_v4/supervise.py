"""Own-process watchdog with existing production GPU flock; no foreign signals."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
SAMPLE_SEQUENCE = 0


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, exclusive=False):
    blob = json.dumps(value, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    else:
        temp = path.with_suffix('.tmp')
        with temp.open('w') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)


def identity(pid):
    base = Path('/proc') / str(pid)
    stat = (base/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, ticks=int(stat[19]), cwd=str((base/'cwd').resolve()),
                argv_sha256=hashlib.sha256((base/'cmdline').read_bytes()).hexdigest())


def parse_gpu(xml, uuid):
    root = ET.fromstring(xml)
    gpu = next(g for g in root.findall('gpu') if g.findtext('uuid') == uuid)
    def mib(text):
        assert text and text.endswith(' MiB'), ('UNKNOWN_GPU_MEMORY',text)
        return int(text.split()[0])
    processes = [dict(pid=int(p.findtext('pid')), mib=mib(p.findtext('used_memory')),
                     type=p.findtext('type')) for p in gpu.findall('./processes/process_info')]
    return dict(uuid=uuid, used_mib=mib(gpu.findtext('./fb_memory_usage/used')), processes=processes)


def snapshot(uuid):
    global SAMPLE_SEQUENCE
    result = subprocess.run(['nvidia-smi','-q','-x'], capture_output=True, text=True, check=True, timeout=5)
    folder = HERE/'resource_raw'
    folder.mkdir(exist_ok=True)
    path = folder/('sample_%06d.json' % SAMPLE_SEQUENCE)
    SAMPLE_SEQUENCE += 1
    write(path, dict(unix=time.time(), xml=result.stdout), exclusive=True)
    parsed = parse_gpu(result.stdout, uuid)
    parsed['raw_path'] = str(path)
    return parsed


def guard(snap, own_pid, budget):
    external = [p for p in snap['processes'] if p['pid'] != own_pid]
    assert all(p['mib'] <= budget['external_per_process_mib'] for p in external), 'EXTERNAL_PROCESS_LOAD'
    assert sum(p['mib'] for p in external) <= budget['external_gpu_mib'], 'EXTERNAL_TOTAL_LOAD'
    own = sum(p['mib'] for p in snap['processes'] if p['pid'] == own_pid)
    assert own*1024**2 <= budget['max_gpu_memory_bytes'], 'OWN_GPU_MEMORY_LIMIT'
    if own_pid is None:
        assert snap['used_mib'] <= budget['external_gpu_mib'], 'INITIAL_GPU_GROSS_LOAD'
    assert max(snap['used_mib'], sum(p['mib'] for p in snap['processes']))*1024**2 <= budget['max_gpu_memory_bytes']+budget['external_gpu_mib']*1024**2, 'GROSS_GPU_LIMIT'


def run():
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    for name, expected in json.loads((HERE/'CODE_SEAL.json').read_text()).items():
        assert digest(HERE/name) == expected, ('SEAL_CHANGED',name)
    assert not (HERE/'SUPERVISOR_RESULT.json').exists(), 'FRESH_SUPERVISOR_ONLY'
    # Reuse the production queue mutex, and inherit into the actual CUDA worker.
    lock = (LINE/'data_pipeline/auto_production_v1/gpu_locks/gpu_2.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = protocol['budget']
    uuid = protocol['gpu_uuid']
    samples = []
    for _ in range(2):
        snap = snapshot(uuid); guard(snap, None, budget); samples.append(snap)
        time.sleep(1)
    write(HERE/'RESOURCE_ADMISSION.json', dict(unix=time.time(), samples=samples,
        protocol_sha256=digest(HERE/'PROTOCOL.json'), supervisor_identity=identity(os.getpid()),
        lock_path=str(Path(lock.name)), holder_borrowed=False, foreign_processes_signalled=0), exclusive=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, Q35N_SUPERVISED_RUN=digest(HERE/'PROTOCOL.json'),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key, subdir in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cache/cuda'),
                        ('HF_HOME','cache/huggingface'),('MPLCONFIGDIR','cache/matplotlib'),('NUMBA_CACHE_DIR','cache/numba'),('TORCHINDUCTOR_CACHE_DIR','cache/inductor'),('TRITON_CACHE_DIR','cache/triton'),('PYTORCH_KERNEL_CACHE_PATH','cache/torch_kernels')]:
        target = HERE/subdir; target.mkdir(parents=True, exist_ok=True); env[key] = str(target)
    command = [str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B','-u',str(HERE/'execute.py'),'--worker']
    started = time.monotonic()
    stop = []
    signal.signal(signal.SIGTERM,lambda *_:stop.append('SUPERVISOR_SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stop.append('SUPERVISOR_SIGINT'))
    child = None
    owned = None
    error = None
    signals_sent = []
    try:
        with (HERE/'worker.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(lock.fileno(),), start_new_session=True)
            # Popen exec is complete, so argv/starttime/cwd can be bound exactly.
            owned = identity(child.pid)
            write(HERE/'WORKER_IDENTITY.json',dict(identity=owned, command=command, gpu_uuid=uuid),exclusive=True)
            while child.poll() is None:
                snap = snapshot(uuid); guard(snap, child.pid, budget)
                elapsed = time.monotonic()-started
                assert elapsed < budget['wall_seconds'], 'OUTER_WALL_BUDGET'
                progress = HERE/'run_0001/PROGRESS.json'
                assert not progress.exists() or time.time()-progress.stat().st_mtime < 600, 'STALE_WORKER_HEARTBEAT'
                if not (HERE/'PROBE.json').exists():
                    assert elapsed < budget['probe_seconds'], 'PROBE_WALL_BUDGET'
                used = sum(p.stat().st_size for p in (HERE/'run_0001').glob('*') if p.is_file())
                assert used < budget['max_new_disk_bytes'], 'RUN_DISK_BUDGET'
                state = dict(status='WORKER_RUNNING', unix=time.time(), elapsed_seconds=elapsed,
                    worker_pid=child.pid, worker_identity=owned, gpu=snap, disk_bytes=used,
                    training_started=(HERE/'run_0001/RUN_ADMISSION.json').exists(),
                    monitor='http://127.0.0.1:18766', foreign_processes_signalled=0)
                write(HERE/'STATUS.json',state)
                with (HERE/'RESOURCE_SAMPLES.jsonl').open('a') as stream:
                    stream.write(json.dumps(state)+'\n')
                if stop:
                    raise RuntimeError(stop[-1])
                time.sleep(10)
    except BaseException as exc:
        error = repr(exc)
        if child is not None and child.poll() is None:
            assert owned is None or identity(child.pid) == owned, 'OWN_CHILD_IDENTITY_CHANGED'
            child.send_signal(signal.SIGTERM); signals_sent.append('SIGTERM_OWN_CHILD')
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                assert owned is None or identity(child.pid) == owned, 'OWN_CHILD_IDENTITY_CHANGED'
                child.kill(); signals_sent.append('SIGKILL_OWN_CHILD'); child.wait(timeout=30)
    finally:
        alive = child is not None and child.poll() is None
        cleanup = False
        final_snapshot = None
        try:
            for _ in range(6):
                final_snapshot = snapshot(uuid)
                cleanup = not alive and (child is None or all(p['pid'] != child.pid for p in final_snapshot['processes']))
                if cleanup:
                    break
                time.sleep(5)
        except BaseException as exc:
            error = error or 'CLEANUP_VERIFICATION:' + repr(exc)
        if not cleanup:
            error = error or 'CLEANUP_NOT_VERIFIED'
        result = dict(status='SUPERVISOR_FINISHED' if error is None and child and child.returncode == 0 else 'FAILED_NO_AUTOMATIC_RETRY',
            unix=time.time(), returncode=child.returncode if child else None, error=error,
            elapsed_seconds=time.monotonic()-started, child_alive=alive,
            own_child_signals=signals_sent, foreign_processes_signalled=0, holder_borrowed=False,
            gpu2_production_restarted=False, cleanup_complete=cleanup, final_gpu_snapshot=final_snapshot)
        write(HERE/'SUPERVISOR_RESULT.json',result,exclusive=True)
        write(HERE/'STATUS.json',result)
        lock.close()
    if error or child.returncode:
        raise SystemExit(1)


def main():
    try:
        run()
    except BaseException as error:
        if not (HERE/'SUPERVISOR_RESULT.json').exists():
            result = dict(status='SUPERVISOR_ADMISSION_OR_RUNTIME_FAILED', error=repr(error),
                          unix=time.time(), cleanup_complete=False, foreign_processes_signalled=0)
            write(HERE/'SUPERVISOR_RESULT.json',result,exclusive=True)
            write(HERE/'STATUS.json',result)
        raise


if __name__ == '__main__':
    main()
