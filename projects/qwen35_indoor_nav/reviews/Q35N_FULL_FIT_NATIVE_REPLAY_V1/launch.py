"""One bounded empty-GPU1 replay; only exact registered own processes may be signaled."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
GPU = 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('native_replay_common', HERE / 'common.py')
helpers = load('frozen_owned_tree_helpers', LINE / 'sft_acceptance/ordinary_history8_paired_train_r1/supervise.py')
own = helpers.own


def gpu_all():
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15))
    return {x.findtext('uuid'): dict(memory_mib=float(x.findtext('fb_memory_usage/used').split()[0]),
            pids=[int(p.findtext('pid')) for p in x.findall('processes/process_info')]) for x in root.findall('gpu')}


def violation(row, allowed, budget):
    if not set(row['gpus'][GPU]['pids']) <= set(allowed):
        return 'FOREIGN_GPU1_CONTEXT'
    if any(set(g['pids']) & set(allowed) for uuid, g in row['gpus'].items() if uuid != GPU):
        return 'OWN_GPU_MAPPING_MISMATCH'
    if row['wall_seconds'] > budget['worker_wall_seconds']:
        return 'WALL_BUDGET'
    if row['gpus'][GPU]['memory_mib'] > budget['gpu_gib'] * 1024:
        return 'GPU_MEMORY_BUDGET'
    if row['rss_bytes'] > budget['cpu_rss_gib'] * 1024**3:
        return 'CPU_RSS_BUDGET'
    if row['output_bytes'] > budget['output_gib'] * 1024**3:
        return 'OUTPUT_BUDGET'
    if row['forward_decisions'] > budget['forward_decisions']:
        return 'FORWARD_BUDGET'
    return None


def main():
    lock = (HERE / 'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (HERE / 'PREFLIGHT.json').exists(), 'NODE_ALREADY_ATTEMPTED'
    c.verify_lock()
    assert c.read(HERE / 'CPU_TEST_RESULT.json')['status'] == 'PASS'
    budget = c.read(HERE / 'REPLAY_PROTOCOL.json')
    assert budget['runtime_allowed'] and budget['id'] == 'Q35N_FULL_FIT_NATIVE_REPLAY_V1'
    initial = gpu_all()
    assert not initial[GPU]['pids'] and initial[GPU]['memory_mib'] < 128, 'GPU1_NOT_EMPTY'
    tmp = LINE / '.tfn1'
    assert tmp.is_dir()
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CONDA_PREFIX'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    for key, relative in dict(HF_HOME='hf', XDG_CACHE_HOME='xdg', TORCH_HOME='torch',
        CUDA_CACHE_PATH='cuda', TORCHINDUCTOR_CACHE_DIR='inductor', TRITON_CACHE_DIR='triton').items():
        path = HERE / 'cache' / relative
        assert path.is_dir()
        env[key] = str(path)
    started = time.monotonic()
    c.write(HERE / 'PREFLIGHT.json', dict(unix=time.time(), gpu=initial, budget=budget,
        source_lock_sha256=c.sha(HERE / 'SOURCE_LOCK.json'), holders_released=[]), True)
    child = None
    guard = None
    seen = {}
    error = None
    signals = []
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, _: stop.append(s))

    def observe():
        if child is not None:
            for pid in helpers.tree(child.pid):
                value = own.inspect(pid)
                if value and value['state'] not in ('Z', 'X'):
                    identity = own.stable(value)
                    assert pid not in seen or seen[pid]['start'] == identity['start'], 'OWNED_TREE_PID_REUSE'
                    seen[pid] = identity

    def live():
        rows = {}
        for pid, identity in seen.items():
            value = own.inspect(pid)
            if value and value['state'] not in ('Z', 'X') and own.stable(value) == identity:
                rows[pid] = identity
        return rows

    with (HERE / 'worker.log').open('x') as log:
        try:
            argv = [str(LINE / '.envs/q35n_qwen_g2_v1/bin/python3'), '-I', '-B', str(HERE / 'worker.py')]
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            guard = own.OwnedChild(child, argv, ROOT)
            c.write(HERE / 'PROCESS.json', dict(**guard.identity, unix=time.time()), True)
            while guard.check() != 'exited':
                observe()
                active = live()
                progress = HERE / 'run_001/PROGRESS.json'
                value = c.read(progress) if progress.exists() else {}
                size, misses = helpers.scan.tree_size(HERE)
                row = dict(unix=time.time(), wall_seconds=time.monotonic() - started,
                    gpus=gpu_all(), owned_pids=sorted(active),
                    rss_bytes=sum(helpers.rss(pid) for pid in active), output_bytes=size,
                    tree_scan_transient_misses=misses,
                    forward_decisions=value.get('total_forward_decisions', 0))
                reason = 'LAUNCHER_SIGNAL' if stop else violation(row, active, budget)
                row['stop_reason'] = reason
                c.write(HERE / 'RESOURCE.json', row)
                c.append(HERE / 'RESOURCE.jsonl', row)
                assert reason is None, reason
                time.sleep(3)
            assert child.returncode == 0, 'WORKER_EXIT:' + str(child.returncode)
            result = c.read(HERE / 'run_001/REPLAY_RESULT.json')
            assert result['status'] == 'COMPLETE_PENDING_CPU_PARITY' and result['total_forward_decisions'] == 5519
        except BaseException as exc:
            error = repr(exc)
            c.write(HERE / 'LAUNCH_FAILURE.json', dict(unix=time.time(), error=error,
                    traceback=traceback.format_exc()), True)
        finally:
            try:
                observe()
            except BaseException as exc:
                error = error or repr(exc)
            cleanup = guard.cleanup(timeout=15) if guard else dict(exited=child is None or child.poll() is not None,
                exit_code=child.poll() if child else None, signals=[], cleanup_error='NO_REGISTERED_CHILD' if child else None)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid, identity in seen.items():
                    if child and pid == child.pid:
                        continue
                    try:
                        if helpers.signal_known(identity, sig):
                            signals.append(dict(pid=pid, signal=int(sig)))
                    except BaseException as exc:
                        error = error or repr(exc)
                if sig == signal.SIGTERM:
                    time.sleep(2)
            drain_deadline = min(started + budget['wall_seconds'], time.monotonic() + 30)
            remaining = sorted(live())
            after = {}
            try:
                while True:
                    after = gpu_all()
                    remaining = sorted(live())
                    if not remaining and not after[GPU]['pids'] or time.monotonic() >= drain_deadline:
                        break
                    time.sleep(1)
            except BaseException as exc:
                error = error or repr(exc)
                after = {}
            clean = cleanup['exited'] and not remaining and GPU in after and not after[GPU]['pids']
            if time.monotonic() - started > budget['wall_seconds']:
                error = error or 'TOTAL_WALL_BUDGET'
            c.write(HERE / 'LAUNCH_RESULT.json', dict(status='COMPLETE_TRANSPORT_ONLY' if error is None and clean else 'FAILED',
                unix=time.time(), error=error, cleanup=cleanup, descendant_signals=signals,
                remaining=remaining, gpu_after=after, wall_seconds=time.monotonic() - started,
                other_processes_signaled=[], holders_released=[], scientific_result=None), True)
    if error or not clean:
        raise RuntimeError(error or 'UNCLEAN_EXIT')


if __name__ == '__main__':
    main()
