"""Single-owner, finite, freshness-aware supervised continuation; no foreign signals."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
PY = LINE / '.envs/q35n_qwen_g2_v1/bin/python3'
OUT = HERE / 'formal'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    tmp = path if exclusive else path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    if not exclusive:
        os.replace(tmp, path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def proc(pid):
    p = Path('/proc') / str(pid)
    try:
        s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(s[1]), start=int(s[19]), state=s[0],
                    argv=(p / 'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(root, known):
    rows = [proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    ids = {root} | set(known)
    while True:
        expanded = ids | {p['pid'] for p in rows if p and p['ppid'] in ids}
        if expanded == ids:
            break
        ids = expanded
    for p in rows:
        if p and p['pid'] in ids:
            known.setdefault(p['pid'], p)


def signal_own(known, signum):
    for pid, old in sorted(known.items()):
        live = proc(pid)
        if live and live['state'] not in ('Z', 'X') and all(live[k] == old[k] for k in ('start', 'argv')):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


def cleanup(process, known):
    descendants(process.pid, known)
    signal_own(known, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not any((p := proc(pid)) and p['state'] not in ('Z', 'X') for pid in known):
            break
        time.sleep(.5)
    signal_own(known, signal.SIGKILL)
    process.wait(timeout=10)


def latest_checkpoint(default):
    paths = [default] + list(OUT.glob('attempt_*/checkpoint_*.pt'))
    valid = []
    for path in paths:
        receipt = read(Path(str(path) + '.json'))
        if receipt and sha(path) == receipt['sha256']:
            valid.append((receipt['cursor']['updates'], path.stat().st_mtime, path, receipt))
    assert valid, 'NO_VALID_CHECKPOINT'
    return max(valid, key=lambda item: item[:2])[2:]


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'RUNNER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(HERE / 'PROTOCOL_FILESTORE.json')
    for name, digest in protocol['code_sha256'].items():
        assert sha(HERE / name) == digest, 'CODE_CHANGED:' + name
    accounting = protocol['accounting']
    charge_base = accounting['initial_charged_decisions']
    resume = Path(protocol['resume_from']['path'])
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _: stop.append(s))
    attempts = []
    status = 'STARTING'
    for number in range(1, 4):
        if stop or time.time() >= accounting['deadline_unix'] or charge_base >= protocol['budget']['max_decisions']:
            break
        resume, receipt = latest_checkpoint(resume)
        run = OUT / ('attempt_%03d' % number)
        run.mkdir(exist_ok=False)
        write(run / 'ATTEMPT_ACCOUNTING.json', dict(resume_sha256=receipt['sha256'],
              charged_base_decisions=charge_base, deadline_unix=accounting['deadline_unix']), True)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=','.join(protocol['gpus']), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TOKENIZERS_PARALLELISM='false',
                   TRITON_CACHE_DIR=str(LINE / 'runtime/cache/triton_sync_recovery_v1'),
                   TORCHINDUCTOR_CACHE_DIR=str(LINE / 'runtime/cache/inductor_sync_recovery_v1'))
        argv = [str(PY), '-I', '-B', str(HERE / 'launcher.py'),
                str(HERE / 'train_filestore.py'), '--protocol', str(HERE / 'PROTOCOL_FILESTORE.json'),
                '--run-dir', str(run), '--resume', str(resume)]
        started = time.time()
        known = {}
        reason = None
        with (run / 'train.log').open('xb') as log:
            process = subprocess.Popen(argv, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    descendants(process.pid, known)
                    progress = read(run / 'PROGRESS.json')
                    age = time.time() - progress.get('unix', started)
                    limit = 120 if progress else 600
                    status = 'TRAINING' if progress and age < limit else ('STARTING' if not progress else 'STALLED')
                    write(OUT / 'STATUS.json', dict(status=status, unix=time.time(), run_dir=str(run),
                          attempt=number, launcher_pid=process.pid, progress_age_seconds=age,
                          latest_updates=progress.get('cursor', {}).get('updates', receipt['cursor']['updates']),
                          resume=str(resume), deadline_unix=accounting['deadline_unix'],
                          max_attempts=3, monitor='http://127.0.0.1:18767',
                          model_navigation_gain_verified=False))
                    if stop or age > limit or time.time() >= accounting['deadline_unix']:
                        reason = 'USER_OR_LEASE_SIGNAL' if stop else ('NO_PROGRESS_TIMEOUT' if age > limit else 'WALL_BUDGET')
                        break
                    time.sleep(3)
            finally:
                cleanup(process, known)
        result = read(run / 'RESULT.json')
        progress = read(run / 'PROGRESS.json')
        observed = max(charge_base, result.get('charged_compute_decisions', 0),
                       progress.get('cumulative_compute', {}).get('decisions', 0))
        normal = process.returncode == 0 and result.get('status') in ('EPOCHS_COMPLETED', 'STOPPED')
        reserve = 0 if normal else protocol['log_every_updates'] * protocol['max_global_batch_decisions']
        charge_base = observed + reserve
        attempt = dict(number=number, exit=process.returncode, reason=reason, started=started,
                       ended=time.time(), result=result, resume=str(resume),
                       observed_charged=observed, unobserved_tail_reserve=reserve,
                       charged_next_attempt=charge_base)
        attempts.append(attempt)
        write(run / 'ATTEMPT_RESULT.json', attempt, True)
        if normal or stop or reason == 'WALL_BUDGET':
            status = result.get('status', 'STOPPED')
            break
        status = 'RETRY_PENDING'
    else:
        status = 'FAILED_ATTEMPTS_EXHAUSTED'
    write(OUT / 'SUPERVISOR_RESULT.json', dict(status=status, attempts=attempts,
          charged_decisions=charge_base, unix=time.time(), scientific_pass=False), True)
    write(OUT / 'STATUS.json', dict(status=status, unix=time.time(), run_dir=str(run) if attempts else None,
          charged_decisions=charge_base, attempts=len(attempts), model_navigation_gain_verified=False))
    if status == 'FAILED_ATTEMPTS_EXHAUSTED':
        raise RuntimeError(status)


if __name__ == '__main__':
    main()
