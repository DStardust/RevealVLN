"""Supervised resilient training for ordinary baseline v3 (external-signal-proof).

The platform has repeatedly delivered unattributed SIGTERM to training processes
(v5, v6, and v3 run_0001 update 3687). This supervisor runs the bounded segment
as resume-linked attempts: each attempt continues from the latest hash-verified
checkpoint. Global attempts cap and the protocol budgets bound everything; the
cumulative decisions account is summed across attempts from receipts.
"""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
TORCHRUN = LINE / '.envs/q35n_qwen_g2_v1/bin/torchrun'

MAX_ATTEMPTS = 24
GPU_UUIDS = ('GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a,GPU-e458147b-4739-d22a-e764-50743cff4a11,'
             'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b')


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def latest_checkpoint(run_dir):
    receipts = sorted(run_dir.glob('checkpoint_*.pt.json'))
    for receipt in reversed(receipts):
        ckpt = receipt.with_suffix('')
        ckpt = Path(str(receipt)[:-5])  # strip .json
        if ckpt.is_file():
            data = json.loads(receipt.read_text())
            if sha256(ckpt) == data['sha256']:
                return ckpt, data
    return None, None


def main():
    run_dir = HERE / 'formal' / 'run_0001'
    protocol = HERE / 'PROTOCOL_3GPU_SEG2.json'
    log_path = HERE / 'formal' / 'supervise_v3.log'
    attempts = []
    started = time.monotonic()
    with log_path.open('a') as log:
        while len(attempts) < MAX_ATTEMPTS:
            resume = None
            if run_dir.is_dir() and (run_dir / 'PROGRESS.jsonl').exists():
                ckpt, receipt = latest_checkpoint(run_dir)
                if ckpt is not None:
                    resume = ckpt
            argv = [str(TORCHRUN), '--standalone', '--nproc_per_node=3',
                    str(HERE / 'train.py'), '--protocol', str(protocol),
                    '--run-dir', str(run_dir)]
            if resume is not None:
                argv += ['--resume', str(resume)]
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS, HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       NCCL_IB_DISABLE='1')
            attempt_started = time.time()
            # No separate session: the lease driver's group cleanup must reach
            # the whole attempt tree on timeout.
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
            attempt = dict(n=len(attempts) + 1, exit=code, unix_started=attempt_started,
                           unix_ended=time.time(), resumed_from=str(resume) if resume else None)
            attempts.append(attempt)
            log.write(json.dumps(attempt) + '\n')
            log.flush()
            result = json.loads((run_dir / 'RESULT.json').read_text()) if (run_dir / 'RESULT.json').is_file() else None
            conts = sorted(run_dir.glob('RESULT_CONTINUATION_*.json'))
            last = json.loads(conts[-1].read_text()) if conts else result
            if last and last.get('status') == 'EPOCHS_COMPLETED':
                break
            if code == 0 and last and last.get('status') == 'STOPPED' and \
                    any(str(s).startswith('BUDGET:') for s in last.get('stop', [])):
                break  # budget exhausted: not a crash, do not retry
        summary = dict(unix=time.time(), attempts=attempts, total_wall=time.monotonic() - started,
                       completed=bool(last and last.get('status') == 'EPOCHS_COMPLETED'),
                       budget_stopped=bool(last and last.get('status') == 'STOPPED'
                                           and any(str(s).startswith('BUDGET:') for s in last.get('stop', []))),
                       attempts_cap=MAX_ATTEMPTS)
        with (HERE / 'formal' / 'SUPERVISOR_RESULT.json').open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
