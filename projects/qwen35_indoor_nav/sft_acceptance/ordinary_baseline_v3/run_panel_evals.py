"""Panel evaluation driver: evaluate selected v3 checkpoints on the frozen FIT panel.

Runs inside a GPU3 lease after a training segment. Each eval is a separate
process (kernel binding is frozen at import); results land in
<run>/panel_evals/EVAL_<checkpoint>.json. Read-only over the training run.
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


def main():
    run_dir = HERE / 'formal' / 'run_0001'
    out = run_dir / 'panel_evals'
    out.mkdir(exist_ok=False)
    spec_path = HERE / 'PANEL_EVAL.json'
    spec = json.loads(spec_path.read_text())
    checkpoints = []
    # first / mid / latest of epoch 0 + final
    receipts = sorted(run_dir.glob('checkpoint_*.pt.json'))
    require(receipts, 'NO_CHECKPOINTS')
    epoch0 = [r for r in receipts]
    picks = [epoch0[0], epoch0[len(epoch0) // 2], epoch0[-1]]
    for receipt in picks:
        ckpt = Path(str(receipt)[:-5])
        import hashlib as _h
        h = _h.sha256(ckpt.read_bytes()).hexdigest()
        require(h == json.loads(receipt.read_text())['sha256'], 'CHECKPOINT_HASH:' + ckpt.name)
        checkpoints.append(ckpt)
    results = []
    for ckpt in checkpoints:
        log = (out / ('eval_%s.log' % ckpt.stem)).open('w')
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        process = subprocess.Popen(
            [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'eval.py'),
             '--eval-protocol', str(spec_path), '--checkpoint', str(ckpt), '--out', str(out)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        while True:
            code = process.poll()
            if code is not None:
                require(code == 0, 'EVAL_EXIT_%s_%d' % (ckpt.stem, code))
                break
            require(time.monotonic() < deadline, 'EVAL_DEADLINE:' + ckpt.stem)
            time.sleep(2)
        log.close()
        result = json.loads((out / ('EVAL_%s.json' % ckpt.stem)).read_text())
        results.append(dict(checkpoint=ckpt.name, ce=result['ce'], macro_recall=result['macro_recall'],
                            action_recall=result['action_recall'],
                            accuracy_minus_always_forward=result['accuracy_minus_always_forward']))
    summary = dict(unix=time.time(), panel=spec['panel_sha256'], kernel=spec['kernel'],
                   results=results)
    with (out / 'PANEL_EVAL_SUMMARY.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=1)[:2000])


if __name__ == '__main__':
    main()
