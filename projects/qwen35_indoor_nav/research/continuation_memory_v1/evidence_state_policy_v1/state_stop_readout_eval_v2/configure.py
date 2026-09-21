"""Freeze the evaluation-only correction before any new autonomous rollout."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    source=PARENT/'state_stop_readout_v1';cfg=read(source/'PROTOCOL.json')
    cfg.update(version='STATE_STOP_READOUT_EVAL_V2',new_training_updates=0,source_run=str(source/'runs/repair_001'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu_session_hours=8,
        previous_local_gate='Preserved at V1: it counted teacher STOP disagreements as false stops. This diagnostic is not a task-semantic failure test.',
        admission='All six existing finals, no selection. Correctness only gates execution; metric tradeoffs reported after full paired DEV. No automatic adoption.',
        new_optimization_allowed=False,repair_updates=1200,
        loss='Not used: evaluation only. Frozen readout checkpoint from first1200-step run, unchanged.',
        authorization='Existing user V5 permits measured bounded shared-GPU development and separates completion/benefit/adoption. Latest user permits localized repair and continuation; no new architecture training in this version.')
    cfg.pop('local_gate',None)
    immutable(HERE/'PROTOCOL.json',cfg)
    paths={LINE/p for p in read(source/'SOURCE_LOCK.json')['files']};paths.update(HERE.glob('*.py'))
    paths.update([HERE/'PROTOCOL.json',HERE/'index.html',source/'semantic_stop_audit_v1.py',source/'runs/repair_001/SEMANTIC_STOP_AUDIT.json'])
    immutable(HERE/'SOURCE_LOCK.json',dict(source_commit=cfg['source_commit'],files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)}))
    print(dict(version=cfg['version'],new_updates=0,planned=768))

if __name__=='__main__':main()
