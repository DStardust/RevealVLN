"""GPU3 acceptance driver: panel (fallback+fla) -> compare -> resume pair -> throughput.

Runs inside a lease with exactly one GPU visible. All heavy phases are
subprocesses of the v3 env interpreter; this driver only orchestrates and
applies the frozen gates from acceptance/ACCEPTANCE.json.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ENV_PYTHON = LINE / '.envs/q35n_qwen_g2_v1/bin/python3'
GPU_UUID = 'GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def run(name, argv, timeout=1500):
    log = (HERE / 'acceptance' / ('gpu3_%s.log' % name)).open('w')
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


def main():
    acceptance = json.loads((HERE / 'acceptance/ACCEPTANCE.json').read_text())
    require(acceptance['id'] == 'Q35N_ORDINARY_BASELINE_V3_ACCEPTANCE', 'ACCEPTANCE_ID')
    for name, digest in acceptance['code_sha256'].items():
        path = HERE / name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'CODE_CHANGED:' + name)
    for name, digest in acceptance['derived_protocols_sha256'].items():
        path = HERE / acceptance['derived_protocols'][name]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'DERIVED_PROTOCOL_CHANGED:' + name)
    out = HERE / 'acceptance' / 'run_gpu3'
    out.mkdir(parents=True, exist_ok=True)
    acc_path = HERE / 'acceptance/ACCEPTANCE.json'

    if (out / 'PANEL_COMPARISON.json').is_file():
        panel = json.loads((out / 'PANEL_COMPARISON.json').read_text())
    else:
        run('panel_fallback', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'accept.py'),
                               '--acceptance', str(acc_path), '--phase', 'panel',
                               '--kernel', 'fallback', '--out', str(out)])
        run('panel_fla', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'accept.py'),
                          '--acceptance', str(acc_path), '--phase', 'panel',
                          '--kernel', 'fla', '--out', str(out)])
        run('panel_compare', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'accept.py'),
                              '--acceptance', str(acc_path), '--phase', 'panel_compare',
                              '--run', str(out)])
        panel = json.loads((out / 'PANEL_COMPARISON.json').read_text())
    kernel = 'fla' if panel['gates']['status'] == 'PASS' else 'fallback'

    proto_resume = HERE / 'acceptance/proto_resume_fallback.json'
    run('resume_a', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                     '--protocol', str(proto_resume), '--run-dir', str(out / 'resume_a')],
        timeout=1200)
    ckpt6 = out / 'resume_a' / 'checkpoint_000000006.pt'
    require(ckpt6.is_file(), 'RESUME_A_CHECKPOINT6_MISSING')
    (out / 'resume_b').mkdir()
    run('resume_b', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                     '--protocol', str(proto_resume), '--run-dir', str(out / 'resume_b'),
                     '--resume', str(ckpt6)], timeout=1200)
    import torch
    a12 = torch.load(out / 'resume_a' / 'checkpoint_000000012.pt', map_location='cpu',
                     weights_only=True)
    b12 = torch.load(out / 'resume_b' / 'checkpoint_000000012.pt', map_location='cpu',
                     weights_only=True)
    require(a12['cursor']['updates'] == b12['cursor']['updates'] == 12, 'RESUME_CURSOR')
    exact = all(torch.equal(a12['trainable'][k], b12['trainable'][k]) for k in a12['trainable'])
    resume_gate = dict(fallback_pair_trainables_exact=exact,
                       note='deterministic fallback path; fla-mode continuity checked in throughput run')

    proto_thr = HERE / ('acceptance/proto_throughput_%s.json' % kernel)
    thr_dir = out / ('throughput_%s' % kernel)
    throughput_result = None
    try:
        run('throughput', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                           '--protocol', str(proto_thr), '--run-dir', str(thr_dir)], timeout=1800)
    except ValueError as exc:
        throughput_result = dict(kernel=kernel, failed=repr(exc))
    if throughput_result is None:
        events = [json.loads(x) for x in (thr_dir / 'PROGRESS.jsonl').read_text().splitlines()
                  if x.strip()]
        started = events[0]['unix']
        steady = [e for e in events if e['unix'] - started >= 120]
        require(len(steady) >= 2, 'STEADY_WINDOW_TOO_SHORT')
        steady_tps = ((steady[-1]['cumulative_compute']['decisions']
                       - steady[0]['cumulative_compute']['decisions'])
                      / (steady[-1]['unix'] - steady[0]['unix']))
        sanity = acceptance['throughput_sanity_single_card_min']
        throughput_result = dict(kernel=kernel, events=len(events),
                                 steady_decisions_per_second=steady_tps,
                                 sanity_min=sanity, sanity_pass=steady_tps >= sanity)
    if throughput_result.get('failed') and kernel == 'fla':
        # FLA selected but failed at runtime: record, no silent kernel switch.
        pass
    elif throughput_result.get('failed'):
        fb1024 = HERE / 'acceptance/proto_throughput_fallback_1024.json'
        thr_dir2 = out / 'throughput_fallback_1024'
        try:
            run('throughput_fallback_1024', [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'train.py'),
                                             '--protocol', str(fb1024), '--run-dir', str(thr_dir2)],
                timeout=1800)
            events = [json.loads(x) for x in (thr_dir2 / 'PROGRESS.jsonl').read_text().splitlines()
                      if x.strip()]
            started = events[0]['unix']
            steady = [e for e in events if e['unix'] - started >= 120]
            steady_tps = ((steady[-1]['cumulative_compute']['decisions']
                           - steady[0]['cumulative_compute']['decisions'])
                          / (steady[-1]['unix'] - steady[0]['unix']))
            throughput_result['fallback_1024'] = dict(steady_decisions_per_second=steady_tps)
        except ValueError as exc:
            throughput_result['fallback_1024'] = dict(failed=repr(exc))
    summary = dict(unix=time.time(),
                   panel=panel['gates'], kernel_selected=kernel,
                   resume=resume_gate,
                   throughput=throughput_result,
                   overall=None)
    thr_ok = bool(throughput_result.get('sanity_pass'))
    summary['overall'] = ('PASS' if panel['gates']['status'] == 'PASS' and exact and thr_ok
                          else 'FAIL')
    with (out / 'GPU3_ACCEPTANCE_SUMMARY.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(overall=summary['overall'], kernel=kernel, resume_exact=exact,
                          throughput=throughput_result)))


if __name__ == '__main__':
    main()
