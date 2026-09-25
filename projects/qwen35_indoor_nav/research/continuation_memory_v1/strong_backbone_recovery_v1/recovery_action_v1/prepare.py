"""Register TRAIN-only recovery demonstrations and a matched architecture test."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as u
from transfer_pipeline import records


def cutoff(trace, maximum=64):
    steps = [r['environment_step'] for r in trace if r['event'] == 'generation']
    return max(t for t in steps if t <= maximum)


def main(run):
    if run.exists():
        raise ValueError('RUN_EXISTS')
    source = HERE.parent/'intervention_v2/runs/recovery_001/train'
    groups = records(source, 'evaluation', True)
    manifest = u.read(source/'DATA_MANIFEST.json')['episodes']
    assert len(groups) == len(manifest) == 800
    failed = [e for e in manifest if not groups[e['id']]['outcomes']['NATIVE']['success']]
    kept = []
    for partition, count in [('FIT', 128), ('DEV', 32)]:
        candidates = [e for e in manifest if e['partition'] == partition and e['variant'] == 'natural'
                      and groups[e['id']]['outcomes']['NATIVE']['success']]
        candidates.sort(key=lambda e: hashlib.sha256(('preserve42:'+e['route_family']).encode()).hexdigest())
        assert len(candidates) >= count
        kept += candidates[:count]
    rows = []
    for e in failed + kept:
        g = groups[e['id']]
        trace_path = Path(g['path']).parent/'NATIVE/TRACE.jsonl'
        assert u.sha(trace_path) == g['trace_hashes']['NATIVE']
        import json
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        kind = 'RECOVERY' if e in failed else 'PRESERVATION'
        rows.append(dict(e, id=len(rows), source_id=e['id'], kind=kind,
                         cutoff=cutoff(trace) if kind == 'RECOVERY' else None,
                         source_trace=str(trace_path), source_trace_sha256=u.sha(trace_path),
                         config_sha256=u.sha(e['config']), native_outcome=g['outcomes']['NATIVE']))
    fit = {e['house'] for e in rows if e['partition'] == 'FIT'}
    dev = {e['house'] for e in rows if e['partition'] == 'DEV'}
    unseen = u.read(source.parent/'unseen/DATA_MANIFEST.json')
    assert not fit & dev and not (fit | dev) & {e['house'] for e in unseen['episodes']}
    assert 'EU6Fwq7SyZv' not in fit | dev
    run.mkdir(parents=True)
    previous = u.read(source/'PROTOCOL.json')
    protocol = dict(id='TRAIN_RECOVERY_ACTION_ARCH_V1', source_commit=previous['source_commit'],
        expected_base_state_sha256=previous['expected_base_state_sha256'], seed=42,
        backbone=previous['backbone'], feature_width=3584, memory_shape=[8,64],
        arms=['CONCAT', 'EVIDENCE'], heads={}, steps=3000, checkpoint_every=200,
        optimizer=dict(name='AdamW', lr=0.0001, weight_decay=0.01, clip_grad_norm=1.0),
        objective='recovery CE + .5*ordinary CE + 5*(ordinary KL + native margin)',
        recovery_class_weight='inverse square root FIT action frequency, mean normalized',
        teacher='Habitat ShortestPathFollower to registered goal, radius 1.8m, executed offline only',
        teacher_prefix='Actual failed native history to last query <=64; RGB exact replay, no teleport',
        teacher_label_scope='Goal-directed R2R recovery actions, not exact language-program labels',
        failed_teacher='Recorded NOT_ADMITTED; no synthetic positive or negative labels',
        actor_boundary='First real action after assistant header; unchanged native four-action decoding',
        runtime='BF16 base, flash_attention_2; FP32 memory and generation scores/addition in training and live',
        gate='NONE: improve action proposals first; existing gate is not trained or loaded',
        planned_collection=len(rows), planned_unseen=200, planned_unseen_episodes=600,
        gpu_indices=list(range(8)), chunk_pairs=16, gpu_session_hours_limit=64,
        wall_hours_limit=24, artifact_limit_gib=32, automatic_score_retries=0,
        phases=['COLLECT', 'FEATURES', 'TRAIN', 'DEV', 'UNSEEN', 'REVIEW'],
        independent_service=True, old_heads_loaded=False, base_updates=0,
        unseen_exposed=True, full_1839=False, positive_result_required_for_continuation=False,
        primary_comparison='EVIDENCE vs CONCAT, paired SR; also each vs same-run NATIVE',
        limitations=['Single seed engineering pilot, not a paper conclusion',
                    'Architecture package comparison does not isolate attention and centering',
                    'Shared recovery data effect is not the architecture contribution'])
    u.write(run/'PROTOCOL.json', protocol)
    u.write(run/'DATA_MANIFEST.json', dict(episodes=rows, source=str(source)))
    u.write(run/'UNSEEN_MANIFEST.json', unseen)
    u.write(run/'SPLIT_AUDIT.json', dict(fit_houses=sorted(fit), dev_houses=sorted(dev),
        counts=dict(Counter(e['partition']+':'+e['kind'] for e in rows)),
        fit_failed_route_families=len({e['route_family'] for e in failed if e['partition']=='FIT'}),
        official_train_only=True, old_memory_EU6_excluded=True, no_old_heads_or_labels=True))
    u.write(run/'STATUS.json', dict(status='PREPARED', phase='COLLECT'))
    u.write(HERE.parent/'RECOVERY_ACTION_RUN.json', dict(path=str(run), run=run.name))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
