"""Register one bounded repair before new optimization; original FIT/DEV only."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    old=read(PARENT/'monotonic_holdout_v1/PROTOCOL.json');training=read(PARENT/'gpu_v1/PROTOCOL.json')
    data=read(CPU/'DATA.json');houses=sorted({f['house'] for f in data['raw_families'] if f['split']=='DEV'})
    if len(houses)!=1:raise ValueError('ORIGINAL_DEV_HOUSE')
    fields=('asset_line_root','torch_python','sim_python','standalone_python','checkpoint','checkpoint_sha256','model_source_sha256',
        'training_protocol_sha256','sample_index_sha256','seed','seeds','model_memory_gib','process_rss_gib','warmup_indices','warmup_repeats','publication_branch','devices','source_models')
    cfg={k:old[k] for k in fields}
    cfg.update(version='STATE_STOP_READOUT_V1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        houses=houses,arms=['MONOTONIC','STOPFIX'],repair_updates=1200,new_training_updates=3600,base_updates=0,original_model_updates=0,
        learning_rate=training['learning_rate'],weight_decay=training['weight_decay'],readout_parameters=113,
        planned_conditions=128,planned_rollouts=768,main_denominator_per_arm=192,control_denominator_per_arm=192,
        gpu_session_hours=12,max_session_hours=3,min_start_gpu_gib=26,min_free_gpu_gib=2,artifact_gib=40,infra_retries=0,
        loss='Inherited action CE + cutoff CE + full native-preservation KL + ordinary CE, all weights 1. State/event losses constant because original model frozen.',
        schedule='Same 1200 FIT schedule and ordinary record pairs per seed as original training; no DEV backward, no new-house data loaded.',
        local_gate=dict(max_ordinary_accuracy_drop=.005,rule='Pooled FIT correct-state missed STOP lower, >=2 seeds better and none worse; DEV missed STOP lower and false STOP no increase; ordinary CHECK accuracy drop <=0.5pp and false STOP no increase.'),
        scope='Exposed original DEV only, controlled stationary SEE2 stopping and reorientation. No natural VLN or independent test claim.',
        primary='STOPFIX minus MONOTONIC safe task_A PASS over full planned N=192 per arm; task_T separate.',
        decision='Complete DEV and positive main difference in >=2 seeds is local development signal only; show collisions, task_T and cost; no automatic adoption.',
        architecture='Frozen original MONOTONIC + Linear(5,16),Tanh,Linear(16,1) residual on STOP only; input predicted four-state probabilities and original STOP-vs-motion margin; zero output initialization.',
        runtime='Original Qwen/processor/2 RGB/8 executed actions and 500-decision safe evaluator. Final method argmax, no STOP override; all real executed actions enter causal history.',
        authorization='User explicitly permits localized repair, standalone execution, progress website, available GPUs and GitHub publication; restore verified own placeholders on exit.')
    immutable(HERE/'PROTOCOL.json',cfg)
    files=read(PARENT/'monotonic_holdout_v1/SOURCE_LOCK.json')['files']
    paths={LINE/p for p in files};paths.update(HERE.glob('*.py'));paths.update([HERE/'PROTOCOL.json',HERE/'index.html'])
    immutable(HERE/'SOURCE_LOCK.json',dict(source_commit=cfg['source_commit'],files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)}))
    print(dict(run='repair_001',new_updates=3600,planned=768,scope=cfg['scope']))

if __name__=='__main__':main()
