"""Bind completed heads to the already fixed unseen manifest, without scores."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u


def prepare(training_run, run):
    import torch
    training_run, run = Path(training_run).resolve(), Path(run).resolve()
    plan = u.read(HERE / 'EVALUATION_PLAN.json')
    source = Path(plan['manifest_path'])
    if u.sha(source) != plan['manifest_sha256']:
        raise ValueError('UNSEEN_MANIFEST_CHANGED')
    config_path = training_run / 'TRAINING_CONFIG.json'
    config = u.read(config_path)
    if config['debug_only'] or config['version'] != 'EXECUTION_ADAPTATION_ALL_TOKENS_V1':
        raise ValueError('DEBUG_OR_UNREGISTERED_MODEL')
    if config['steps'] != 3000 or config['seeds'] != [42, 43, 44] or config['modes'] != ['CURRENT', 'DELTA']:
        raise ValueError('TRAINING_REGISTRATION_CHANGED')
    if config['action_scope'] != 'ALL_ACTION_TOKENS_QUERY_START_MEMORY':
        raise ValueError('TRAINING_RUNTIME_ACTION_SCOPE_MISMATCH')
    files = dict(config['source_files'], **config['input_files'])
    files[str(config_path)] = u.sha(config_path)
    heads = {}
    for seed in config['seeds']:
        for mode in config['modes']:
            arm = f'{mode}_s{seed}'
            final = training_run / arm / 'FINAL.pt'
            result = u.read(final.parent / 'RESULT.json')
            if result['status'] != 'COMPLETE' or result['debug_only'] or u.sha(final) != result['final_sha256']:
                raise ValueError('INCOMPLETE_OR_CHANGED_HEAD')
            saved = torch.load(final, map_location='cpu', weights_only=False)
            expected = dict(config_sha256=u.sha(config_path), seed=seed, mode=mode)
            if saved['step'] != 3000 or saved['binding'] != expected:
                raise ValueError('TRAINED_HEAD_BINDING_CHANGED')
            heads[arm] = dict(path=str(final), sha256=result['final_sha256'], seed=seed, mode=mode,
                              training_config_path=str(config_path), training_config_sha256=u.sha(config_path))
            files[str(final)] = result['final_sha256']
    origin = BASE / 'history_validation_v4/runs/validation_001'
    old = u.read(origin / 'PROTOCOL.json')
    inherited = u.read(origin / 'SOURCE_LOCK.json')['files']
    # Real evaluator/processor/sensor code and fixture bytes remain inherited.
    files.update({p: h for p, h in inherited.items() if p.endswith(('.py', '.yaml', '.json.gz'))})
    manifest = u.read(source)
    if len(manifest['episodes']) != plan['groups']:
        raise ValueError('UNSEEN_DENOMINATOR_CHANGED')
    protocol = {k: old[k] for k in ('source_commit', 'expected_base_state_sha256', 'backbone',
                                  'feature_width', 'memory_shape', 'runtime')}
    protocol.update(id='EXECUTION_ADAPTATION_UNSEEN_V1', heads=heads, seeds=config['seeds'], steps=3000,
        split='val_unseen', planned_groups=plan['groups'], planned_executions=plan['groups'] * 7,
        action_scope=config['action_scope'], native_full_vocabulary_decoding_unchanged=True,
        maximum_decisions=500, active_stop_distance_lt_m=3., base_updates=0,
        primary='Paired DELTA minus CURRENT SR, per seed and house; shared all-token data and action scope',
        secondary='Each new head minus same-session NATIVE; SPL, steps, regressions, STOP, resource cost',
        final_head_selection=False, prior_public_benchmark_exposure=True,
        relation_to_v4='Shared token coverage and execution changes are not DELTA-specific contributions',
        arm_order='NATIVE then six heads rotated by frozen episode id; all seven in one model process')
    for path in HERE.glob('*.py'):
        files[str(path)] = u.sha(path)
    files[str(HERE / 'EVALUATION_PLAN.json')] = u.sha(HERE / 'EVALUATION_PLAN.json')
    for path, sha in files.items():
        if u.sha(path) != sha:
            raise ValueError('SOURCE_OR_ASSET_CHANGED: ' + path)
    run.mkdir(parents=True, exist_ok=False)
    u.write(run / 'PROTOCOL.json', protocol)
    u.write(run / 'DATA_MANIFEST.json', manifest)
    for name in ('PROTOCOL.json', 'DATA_MANIFEST.json'):
        files[str(run / name)] = u.sha(run / name)
    u.write(run / 'SOURCE_LOCK.json', dict(files=files))
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_run, args.run), ensure_ascii=False))
