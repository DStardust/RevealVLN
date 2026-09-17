"""GPU3 acceptance v2 driver: dual-kernel training equivalence + fla resume + throughput.

Order: dual fallback (300 upd @1024) -> dual fla (300 upd @1024) -> dual_compare ->
fla resume continuity pair (@6144) -> fla throughput @6144. Frozen gates from
acceptance/ACCEPTANCE.json acceptance_v2; no threshold edited after observation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ENV_PYTHON = LINE / '.envs/q35n_qwen_g2_v1/bin/python3'
GPU_UUID = 'GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def wait_gpu_clear(timeout=90):
    """A just-exited phase's CUDA context and DataLoader workers can lag; wait
    until the card shows no compute processes before starting the next phase."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = subprocess.check_output(
            ['nvidia-smi', '-i', GPU_UUID, '--query-compute-apps=pid',
             '--format=csv,noheader'], text=True, timeout=15).strip()
        if not out:
            return
        time.sleep(2)
    raise ValueError('GPU_NOT_CLEAR_AFTER_PHASE')


def run(name, argv, timeout=1800):
    log = (HERE / 'acceptance' / ('dual_%s.log' % name)).open('w')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=str(HERE))
    deadline = time.monotonic() + timeout
    try:
        while True:
            code = process.poll()
            if code is not None:
                require(code == 0, 'PHASE_%s_EXIT_%d' % (name.upper(), code))
                wait_gpu_clear()
                return
            require(time.monotonic() < deadline, 'PHASE_DEADLINE:' + name)
            time.sleep(2)
    except BaseException:
        if process.poll() is None:
            import signal as _s
            os.killpg(process.pid, _s.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, _s.SIGKILL)
        raise
    finally:
        log.close()


def steady_tps(run_dir, warmup_seconds=120):
    events = [json.loads(x) for x in (run_dir / 'PROGRESS.jsonl').read_text().splitlines()
              if x.strip()]
    started = events[0]['unix']
    steady = [e for e in events if e['unix'] - started >= warmup_seconds]
    require(len(steady) >= 2, 'STEADY_WINDOW_TOO_SHORT')
    return ((steady[-1]['cumulative_compute']['decisions']
             - steady[0]['cumulative_compute']['decisions'])
            / (steady[-1]['unix'] - steady[0]['unix']))


def main():
    acceptance = json.loads((HERE / 'acceptance/ACCEPTANCE.json').read_text())
    for name, digest in acceptance['code_sha256'].items():
        path = HERE / name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'CODE_CHANGED:' + name)
    for name, digest in acceptance['derived_protocols_sha256'].items():
        path = HERE / acceptance['derived_protocols'][name]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'DERIVED_PROTOCOL_CHANGED:' + name)
    out = HERE / 'acceptance' / 'run_dual'
    out.mkdir(exist_ok=True)
    acc_path = HERE / 'acceptance/ACCEPTANCE.json'

    def phase_dir(name):
        """Skip completed phases; clean only own incomplete (no RESULT) run dirs."""
        path = out / name
        if (path / 'RESULT.json').is_file():
            return None
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        return path

    for kernel in ('fallback', 'fla'):
        target = phase_dir('dual_%s' % kernel)
        if target is not None:
            run('dual_%s' % kernel, [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                                     '--protocol', str(HERE / ('acceptance/proto_dual_%s.json' % kernel)),
                                     '--run-dir', str(target)], timeout=2700)
    if not (out / 'DUAL_COMPARISON.json').is_file():
        run('dual_compare', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'dual_compare.py'),
                             '--acceptance', str(acc_path), '--run', str(out)])
    dual = json.loads((out / 'DUAL_COMPARISON.json').read_text())

    proto_resume = HERE / 'acceptance/proto_resume_fla.json'
    target = phase_dir('resume_fla_a')
    if target is not None:
        run('resume_fla_a', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                             '--protocol', str(proto_resume), '--run-dir', str(target)],
            timeout=1500)
    ckpt6 = out / 'resume_fla_a' / 'checkpoint_000000006.pt'
    require(ckpt6.is_file(), 'RESUME_FLA_A_CHECKPOINT6_MISSING')
    target = phase_dir('resume_fla_b')
    if target is not None:
        target.mkdir()  # train.py requires an existing dir on resume
        run('resume_fla_b', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                             '--protocol', str(proto_resume), '--run-dir', str(target),
                             '--resume', str(ckpt6)], timeout=1500)
    import torch
    b12 = torch.load(out / 'resume_fla_b' / 'checkpoint_000000012.pt', map_location='cpu',
                     weights_only=True)
    resume_gate = dict(reached_update_12=b12['cursor']['updates'] == 12,
                       checkpoint_roundtrip=True,
                       bitwise_equality_not_required='triton kernels (KERNEL_DECISION.json)')

    thr_dir = phase_dir('throughput_fla_6144')
    if thr_dir is not None:
        run('throughput_fla', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                               '--protocol', str(HERE / 'acceptance/proto_throughput_fla.json'),
                               '--run-dir', str(thr_dir)], timeout=1800)
    tps = steady_tps(out / 'throughput_fla_6144')
    sanity = acceptance['throughput_sanity_single_card_min']
    summary = dict(unix=time.time(), dual=dual['gates'], resume_fla=resume_gate,
                   throughput_fla_6144=dict(steady_decisions_per_second=tps, sanity_min=sanity,
                                            sanity_pass=tps >= sanity),
                   overall='PASS' if (dual['gates']['status'] == 'PASS'
                                      and resume_gate['reached_update_12']
                                      and tps >= sanity) else 'FAIL')
    with (out / 'GPU3_DUAL_SUMMARY.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(overall=summary['overall'], fla_steady_tps=round(tps, 3),
                          dual=dual['gates']['status'])))


if __name__ == '__main__':
    main()
