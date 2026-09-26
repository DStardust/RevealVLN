"""Bind actual replay assets and the full recapture denominator before GPU use."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u
from contracts import summarize_coverage


def main(run):
    if run.exists():
        raise ValueError('RUN_ALREADY_EXISTS')
    source = BASE / 'recovery_action_v1/runs/action_001/features'
    old_protocol = u.read(source / 'PROTOCOL.json')
    prior_lock = u.read(source / 'SOURCE_LOCK.json')['files']
    index = HERE.parent / 'token_index'
    index_result = u.read(index / 'RESULT.json')
    for name, digest in index_result['artifact_hashes'].items():
        assert u.sha(index / name) == digest, 'TOKEN_INDEX_CHANGED'
    index_rows = {r['id']: r for r in u.read(index / 'TRAJECTORIES.json')}
    plan_dir = HERE.parent / 'recapture_plan_001'
    plan_result = u.read(plan_dir / 'RESULT.json')
    assert u.sha(plan_dir / 'REQUESTS.json') == plan_result['requests_sha256']
    requests = u.read(plan_dir / 'REQUESTS.json')
    plans = defaultdict(list)
    for request in requests:
        plans[request['trajectory_id']].append(request)
    coverage = summarize_coverage(requests, [])
    files = {}
    rows = []
    for entry in u.read(source / 'DATA_MANIFEST.json')['episodes']:
        trajectory = Path(entry['trajectory'])
        assert u.sha(trajectory) == prior_lock[str(trajectory)], 'PHYSICAL_SOURCE_CHANGED'
        old_cache = Path(index_rows[entry['id']]['original_cache_source']['path'])
        old_trace = old_cache.parent / 'NATIVE/TRACE.jsonl'
        assert u.sha(old_cache) == index_rows[entry['id']]['original_cache_source']['sha256'], 'CACHE_CHANGED'
        assert u.sha(entry['config']) == entry['config_sha256'], 'CONFIG_CHANGED'
        assert old_trace.is_file()
        row = dict(entry, trajectory_sha256=prior_lock[str(trajectory)],
            old_cache=str(old_cache), old_cache_sha256=u.sha(old_cache),
            old_trace=str(old_trace), old_trace_sha256=u.sha(old_trace))
        row['missing_stop'] = sum(a['terminal_stop'] and a['recapture_required']
                                  for p in plans[entry['id']] for a in p['actions'])
        row['missing_actor_positions'] = sum(p['missing_actor_positions'] for p in plans[entry['id']])
        assert entry['partition'] in ('FIT', 'DEV') and plans[entry['id']]
        rows.append(row)
        config = Path(entry['config'])
        datasets = [line.split('data_path:', 1)[1].strip() for line in config.read_text().splitlines() if 'data_path:' in line]
        assert len(datasets) == 1
        for path in (trajectory, old_trace, config, Path(datasets[0])):
            files[str(path)] = u.sha(path)
    assert len(rows) == 334 and coverage['totals']['planned_missing_actor_positions'] == 21396
    assert coverage['totals']['planned_missing_stop_positions'] == 133
    fit = {r['house'] for r in rows if r['partition'] == 'FIT'}
    dev = {r['house'] for r in rows if r['partition'] == 'DEV'}
    assert not fit & dev
    # Golden full trajectories cover a missing FIT STOP, missing DEV STOP and
    # recovery's unknown prefix. They remain part of the one 334-row dataset.
    smoke = [min(r['id'] for r in rows if r['partition'] == split and r['missing_stop'])
             for split in ('FIT', 'DEV')]
    smoke += [min(r['id'] for r in rows if r['kind'] == 'RECOVERY' and r['partition'] == 'FIT')]
    order = smoke + [r['id'] for r in sorted(rows, key=lambda r: (-r['missing_stop'], r['partition'] != 'FIT', r['id']))
                     if r['id'] not in smoke]
    for path in [HERE / n for n in ('prepare.py', 'worker.py', 'contracts.py', 'pipeline.py')] + [
        BASE / n for n in ('common.py', 'action_boundary_v2.py', 'capture_runtime_v3.py', 'memory_v2.py',
                          'transfer_pipeline.py', 'unseen_pipeline.py', 'recovery_action_v1/replay.py')]:
        files[str(path)] = u.sha(path)
    for path, digest in prior_lock.items():
        if Path(path).is_relative_to(u.CODE):
            assert u.sha(path) == digest, 'PUBLIC_BACKBONE_CODE_CHANGED'
            files[path] = digest
    for folder in (u.MODEL, u.MODEL / 'siglip-so400m-patch14-384'):
        for path in folder.glob('*.json'):
            files[str(path)] = u.sha(path)
    for path in (u.PYTHON, u.CODE / 'streamvln/streamvln_eval.py', u.MODEL / 'config.json'):
        assert path.is_file(), str(path)
    protocol = dict(id='REAL_ALL_TOKEN_RECAPTURE_V1', source_commit=old_protocol['source_commit'],
        backbone=old_protocol['backbone'], expected_base_state_sha256=old_protocol['expected_base_state_sha256'],
        source=str(source), planned_trajectories=334, planned_missing=21396, planned_stop=133,
        planned_all_actor_rows=28859, planned_queries=7463, planned_requested_queries=7211,
        fit_houses=sorted(fit), dev_houses=sorted(dev), smoke_ids=smoke, run_order=order,
        priority='Missing terminal STOP trajectories first; FIT and DEV remain separate.',
        source_selection='All 334 frozen actual trajectories; no score-based removal.',
        capture='Raw logits and hidden feature before forcing each actual action token; EOS is never a label.',
        causal_context='Original query_start memory for every token in its chunk; future physical features audit only.',
        replay='Initial reset and all actual original actions; raw RGB and processed generation inputs exact.',
        numeric='Record first-token cache/live feature and logit drift and argmax flips; do not mix old/new actor caches.',
        masks='Preserve original supervised region; unknown prefixes stay unknown. No automatic new training admission.',
        dtype='BF16 frozen backbone, FP32 logits/features', attention='flash_attention_2',
        observation='Original RGB/depth/relative pose/intrinsics and original StreamVLN history; no goal input.',
        seed=42, max_decisions=500, stop_no_extra_observation=True,
        optimizer_updates=0, new_alternative_branch_executions=0,
        final_evaluation=dict(split='val_unseen', dev_is_diagnostic_only=True,
            freeze_list_and_comparisons_before_execution=True,
            same_denominator_for_compared_models=True, public_unseen_previously_exposed=True,
            auto_train_or_evaluate_from_this_capture_job=False),
        wait_for_service='q35n-strong-history-validation-20260926-01.service',
        wait_hours_limit=24, gpu_indices=list(range(8)), chunk_trajectories=16,
        gpu_session_hours_limit=16, capture_wall_hours_limit=12, artifact_limit_gib=20,
        automatic_retries=0, resume_boundary='Completed trajectory in state-sealed session; failed attempts retained.',
        phases=['WAIT_DEPENDENCY', 'GPU_SMOKE', 'CAPTURE', 'REVIEW'],
        interpretation='Feature coverage repair only; not new independent routes, new training, or SR improvement.')
    run.mkdir(parents=True)
    u.write(run / 'PROTOCOL.json', protocol)
    u.write(run / 'DATA_MANIFEST.json', dict(episodes=rows))
    u.write(run / 'REQUESTS.json', requests)
    for path in (run / 'PROTOCOL.json', run / 'DATA_MANIFEST.json', run / 'REQUESTS.json'):
        files[str(path)] = u.sha(path)
    u.write(run / 'SOURCE_LOCK.json', dict(files=files))
    u.write(run / 'PREFLIGHT.json', dict(status='CPU_ASSETS_AND_DENOMINATOR_VERIFIED',
        coverage=coverage, model_loaded=False, gpu_hours=0,
        dependency=protocol['wait_for_service'], fit_dev_house_overlap=False))
    u.write(run / 'STATUS.json', dict(status='PREPARED', phase='WAIT_DEPENDENCY',
        planned_trajectories=334, planned_missing=21396, planned_stop=133,
        recorded_trajectories=0, sealed_trajectories=0, captured_missing=0, captured_stop=0,
        gpu_hours=0, workers=[]))
    print(dict(status='PREPARED', run=str(run), smoke=smoke, denominator=334))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
