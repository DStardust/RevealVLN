"""GPU3/4/5 acceptance driver: DDP parity (1 process vs torchrun 3 ranks).

Runs inside a multi-holder lease. Kernel selection is read from the completed
GPU3 acceptance summary (panel verdict). Frozen thresholds come from
acceptance/ACCEPTANCE.json; no threshold is edited after any measurement.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ENV_DIR = LINE / '.envs/q35n_qwen_g2_v1/bin'
UUIDS = ['GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a',
         'GPU-e458147b-4739-d22a-e764-50743cff4a11',
         'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def run(name, argv, timeout=1500, gpu_subset=None):
    log = (HERE / 'acceptance' / ('ddp_%s.log' % name)).open('w')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpu_subset or UUIDS), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               NCCL_IB_DISABLE='1', NCCL_P2P_LEVEL='SYS')
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
    summary = json.loads((HERE / 'acceptance/run_dual/GPU3_DUAL_SUMMARY.json').read_text())
    # Kernel adoption path (option A): informative dual gates must pass and the
    # ce_band sub-gate must carry the registered invalid-measurement disposition.
    require(summary['dual']['ce_final_pass'] and summary['dual']['recall_pass'],
            'DUAL_INFORMATIVE_GATES_NOT_PASSED')
    require('ce_band_disposition' in acceptance.get('acceptance_v2', {}),
            'CE_BAND_DISPOSITION_MISSING')
    require(summary['resume_fla']['reached_update_12'], 'FLA_RESUME_CONTINUITY_FAILED')
    kernel = 'fla'
    acceptance['kernel'] = kernel
    acc_runtime = HERE / 'acceptance/ACCEPTANCE_RUNTIME_DDP.json'
    with acc_runtime.open('w') as stream:
        json.dump(acceptance, stream, indent=2)
    out = HERE / 'acceptance' / 'run_ddp345'
    if out.is_dir():
        require(not (out / 'DDP345_ACCEPTANCE_SUMMARY.json').exists(), 'DDP345_ALREADY_COMPLETED')
        import shutil
        shutil.rmtree(out)  # own incomplete acceptance dir only
    out.mkdir(parents=True, exist_ok=False)

    run('parity_single', [str(ENV_DIR / 'python3'), '-I', '-B', '-u', str(HERE / 'accept.py'),
                          '--acceptance', str(acc_runtime), '--phase', 'parity_single',
                          '--out', str(out)], gpu_subset=UUIDS[:1])
    try:
        run('parity_multi', [str(ENV_DIR / 'torchrun'), '--standalone', '--nproc_per_node=3',
                             str(HERE / 'accept.py'), '--acceptance', str(acc_runtime),
                             '--phase', 'parity_multi', '--out', str(out)])
    except ValueError as exc:
        # NCCL/torchrun teardown can abort nonzero ranks after rank 0 has safely
        # written and fsynced the dump; accept only with the completion marker.
        require((out / 'PARITY_MULTI_DONE.json').is_file(),
                'PARITY_MULTI_TEARDOWN_AND_NO_RESULT: ' + str(exc))
    run('parity_compare', [str(ENV_DIR / 'python3'), '-I', '-B', '-u', str(HERE / 'accept.py'),
                           '--acceptance', str(acc_runtime), '--phase', 'parity_compare',
                           '--run', str(out)])
    parity = json.loads((out / 'PARITY_COMPARISON.json').read_text())
    result = dict(unix=time.time(), kernel=kernel, parity=parity['gates'],
                  overall='PASS' if parity['gates']['status'] == 'PASS' else 'FAIL')
    with (out / 'DDP345_ACCEPTANCE_SUMMARY.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
