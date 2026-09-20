"""Account for every actual V15 GPU session, including failed infrastructure."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review as r


def main():
    raw=r.LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15'
    runs=sorted(raw.glob('raw_run_*'))+sorted(r.HERE.glob('features_run_*'))+sorted(r.HERE.glob('train_run_*'))+sorted(r.HERE.glob('continuation_run_*'))
    rows=[];metadata=[]
    for run in runs:
        if not run.is_dir():continue
        result=r.c.read(run/'LAUNCH_RESULT.json')
        samples=r.c.records(run/'RESOURCE.jsonl')
        assert not result['foreign_processes_signaled']
        row=dict(run=str(run.relative_to(r.LINE)),status=result['status'],gpu_session_seconds=result['gpu_seconds'],
            sampled_peak_own_gpu_mib=max(s['own_memory_mib'] for s in samples),
            sampled_peak_process_group_rss_bytes=max(s['rss_bytes'] for s in samples),
            sampled_peak_monitored_output_bytes=max(s['output_bytes'] for s in samples),
            minimum_card_free_mib=min(s['selected']['free_mib'] for s in samples),
            samples=len(samples),foreign_processes_signaled=0)
        rows.append(row)
        # Persist available compiled metadata, not binaries or a guessed dispatch.
        for path in sorted((run/'cache/triton').glob('**/*.json')):
            data=r.c.read(path)
            metadata.append(dict(run=run.name,path=str(path.relative_to(run)),sha256=r.c.sha(path),metadata=data))
    forwards=[]
    for run in sorted(r.HERE.glob('continuation_run_*')):
        if not run.is_dir():continue
        prefix=sum(len(r.c.records(p)) for p in (run/'rollouts').glob('*/PREFILL.jsonl'))
        autonomous=sum(len(r.c.records(p)) for p in (run/'rollouts').glob('*/POLICY_STEPS.jsonl'))
        forwards.append(dict(run=run.name,recorded_prefill_forwards=prefix,recorded_autonomous_forwards=autonomous))
    result=dict(sessions=rows,gpu_session_hours=sum(x['gpu_session_seconds'] for x in rows)/3600,
        definition='Sum of launcher wall time while owning at most one GPU, including load/compile/wait/cleanup and failed attempts; not CUDA kernel-active hours.',
        own_vs_card_memory='Own process group memory and whole-card free memory are separate.',
        gpu_memory_limit='Monitor snapshots include compute C and graphics G processes, including the owned Habitat EGL process when reported. Peaks are sampled, not allocator instrumentation.',
        known_completed_raw_decisions=sum(r.c.read(raw/f'raw_run_{i:03d}'/'RESULT.json')['actual_decisions'] for i in range(3,9)),
        interrupted_raw_001_002_decisions=None,interrupted_raw_reason='Only partial traces/progress retained; no fabricated exact execution count.',
        feature_extraction_qwen_forwards=r.c.read(r.HERE/'features_run_001/FEATURE_RESULT.json')['real_qwen_forwards'],
        continuation_recorded_forwards=forwards,base_optimizer_updates=0,memory_optimizer_updates=10800,
        disposable_cpu_test_updates=6,foreign_processes_signaled=0,
        performance_scope='Development runtime on an authorized shared-capable GPU; not real-time robot deployment benchmarking.')
    r.c.write(r.HERE/'RESOURCE_REVIEW.json',result)
    r.c.write(r.HERE/'TRITON_CACHE_METADATA.json',dict(entries=metadata,
        actual_kernel_launch_sequence='unavailable; compilation metadata is not a complete launch trace',
        interpretation='Recorded artifacts only. No autotune scan or causal attribution of numeric drift.'))
    print(dict(gpu_session_hours=result['gpu_session_hours'],sessions=len(rows),triton_metadata=len(metadata)))


if __name__=='__main__':main()
