"""Freeze the single operator change before the first new rollout."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    source=PARENT/'state_stop_readout_eval_v2';cfg=read(source/'PROTOCOL.json')
    cfg.update(version='STATE_STOP_READOUT_POSITIVE_V3',arms=['MONOTONIC','STOPPLUS'],new_training_updates=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary='STOPPLUS minus MONOTONIC safe task_A PASS over planned N=192 per arm; task_T separate.',
        architecture='Identical fixed weights: delta = max(0, frozen_readout(predicted_state, original_MONOTONIC_STOP_margin)); only STOP logit increases.',
        runtime='Original final argmax, Qwen/input/500-decision safe evaluator unchanged. Protects original MONOTONIC STOP, not a raw-Qwen STOP override.',
        decision='Report full paired change, added premature STOP, collisions, retained successes and cost. No automatic adoption; one exposed DEV house only.',
        authorization='User 2026-09-22: quickly implement the proposed one-way STOP correction; use authorized GPUs, standalone execution, existing monitor and GitHub publication. Zero new optimization.',
        new_optimization_allowed=False)
    cfg.pop('previous_local_gate',None)
    immutable(HERE/'PROTOCOL.json',cfg)
    paths={LINE/p for p in read(source/'SOURCE_LOCK.json')['files']};paths.update(HERE.glob('*.py'));paths.update([HERE/'PROTOCOL.json',HERE/'index.html',LINE/'reviews/Q35N_STOP_READOUT_REVIEW_20260922/FIRST_DIVERGENCES.json'])
    immutable(HERE/'SOURCE_LOCK.json',dict(source_commit=cfg['source_commit'],files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)}))
    print(dict(version=cfg['version'],new_updates=0,planned=768))

if __name__=='__main__':main()
