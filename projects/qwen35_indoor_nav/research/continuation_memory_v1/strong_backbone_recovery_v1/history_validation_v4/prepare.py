"""Freeze all remaining route families and existing heads without reading scores."""
import gzip
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(BASE), str(BASE/'recovery_confirmation_v2')]
import common as u
from prepare_unseen import pick

SOURCE = BASE/'recovery_confirmation_v2/runs/confirmation_001'
RUN = HERE/'runs/validation_001'


def route_key(episode):
    return (episode.get('house') or episode['scene_id'].split('/')[-2], str(episode['trajectory_id']))


def freeze(run):
    previous = u.read(SOURCE/'unseen/SOURCE_LOCK.json')['files']
    files = {p: h for p, h in previous.items() if p.endswith('.py')}
    for p, h in files.items():
        assert u.sha(p) == h, 'PREVIOUS_RUNTIME_CHANGED: '+p
    own = list(HERE.glob('*.py')) + list((run/'fixtures').iterdir())
    own += [run/n for n in ('PROTOCOL.json', 'DATA_MANIFEST.json', 'SPLIT_AUDIT.json')]
    own += [Path(v['path']) for v in u.read(run/'PROTOCOL.json')['heads'].values()]
    for p in own:
        files[str(p)] = u.sha(p)
    lock = run/'SOURCE_LOCK.json'
    if lock.exists():
        assert u.read(lock)['files'] == files, 'SOURCE_LOCK_CHANGED'
    else:
        u.write(lock, dict(files=files, reused_worker=str(BASE/'recovery_confirmation_v2/worker.py')))


def main():
    run = RUN
    run.mkdir(parents=True, exist_ok=False)
    source = u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz'
    raw = json.loads(gzip.decompress(source.read_bytes()))['episodes']
    assert len(raw) == 1839
    previous = u.read(SOURCE/'unseen/PROTOCOL.json')
    old_audit = u.read(BASE/'runs/unseen_001/SPLIT_AUDIT.json')
    training = u.read(BASE/'recovery_action_v1/runs/action_001/SPLIT_AUDIT.json')
    excluded_houses = set(old_audit['excluded_memory_fit_houses'] + training['fit_houses'] + training['dev_houses'])
    manifests = subprocess.run(['rg', '--files', '-g', 'DATA_MANIFEST.json', '.'], cwd=BASE,
                               check=True, capture_output=True, text=True).stdout.splitlines()
    excluded = set()
    evidence = []
    for name in sorted(manifests):
        path = BASE/name
        if path.is_relative_to(HERE):
            continue
        entries = u.read(path)['episodes']
        keys = set()
        for e in entries:
            if 'trajectory_id' not in e:
                config = Path(e['config']).read_text()
                paths = [line.split('data_path:', 1)[1].strip() for line in config.splitlines() if 'data_path:' in line]
                assert len(paths) == 1
                dataset = json.loads(gzip.decompress(Path(paths[0]).read_bytes()))['episodes']
                assert len(dataset) == 1 and str(dataset[0]['episode_id']) == str(e['episode_id'])
                e = dataset[0]
            keys.add(route_key(e))
        excluded.update(keys)
        evidence.append(dict(path=str(path), sha256=u.sha(path), route_count=len(keys)))
    eligible = {route_key(e) for e in raw if route_key(e) not in excluded and route_key(e)[0] not in excluded_houses}
    # Exhaust the eligible route families; never add paraphrases to hit a round N.
    rows, selection = pick(raw, len(eligible), 1209, excluded_houses, excluded)
    assert {route_key(e) for e in rows} == eligible
    vocab_fixture = u.read(u.ORIGIN/'FIXTURE_MANIFEST.json')[0]
    vocab_path = next(k for k in vocab_fixture['files'] if k.endswith('.json.gz'))
    vocab = json.loads(gzip.decompress((u.ASSETS/vocab_path).read_bytes()))['instruction_vocab']
    template = (u.CODE/'config/vln_r2r.yaml').read_text()
    fixtures = run/'fixtures'
    fixtures.mkdir()
    entries = []
    for index, ep in enumerate(rows):
        dataset = fixtures/f'{index:03d}.json.gz'
        config = fixtures/f'{index:03d}.yaml'
        assert (u.ASSETS/'third_party/ETP-R1/data/scene_datasets'/ep['scene_id']).is_file()
        dataset.write_bytes(gzip.compress(json.dumps(dict(episodes=[ep], instruction_vocab=vocab)).encode(), mtime=0))
        text = template.replace('scenes_dir: data/scene_datasets/', f'scenes_dir: {u.ASSETS}/third_party/ETP-R1/data/scene_datasets/')
        text = text.replace('data_path: data/datasets/r2r/{split}/{split}.json.gz', f'data_path: {dataset}')
        text = text.replace('split: val_seen', 'split: val_unseen').replace('habitat:\n', 'habitat:\n  seed: 42\n', 1)
        config.write_text(text)
        entries.append(dict(id=index, episode_id=ep['episode_id'], trajectory_id=ep['trajectory_id'],
                            house=route_key(ep)[0], config=str(config), config_sha256=u.sha(config),
                            instruction=ep['instruction']['instruction_text']))
    keep = ('source_commit', 'expected_base_state_sha256', 'backbone', 'feature_width', 'memory_shape',
            'heads', 'models', 'seeds', 'steps', 'runtime', 'actor_boundary', 'local_equation')
    protocol = {k: previous[k] for k in keep}
    protocol.update(id='Q35N_HISTORY_VALIDATION_V4', scope_revision='Explicit continuation of WHOLE_VLN_PROGRAM_V1; old freezes unchanged',
                    source_run=str(SOURCE), split='OFFICIAL_VAL_UNSEEN_REMAINING_ROUTE_FAMILIES',
                    phases=['CPU_MECHANISM', 'FROZEN_UNSEEN_EXTENSION', 'REVIEW'],
                    planned_groups=len(entries), planned_executions=len(entries)*7,
                    capture_only=False, optimizer_updates=0, base_updates=0, heads_frozen=True,
                    selection='All remaining eligible physical routes; one hash-selected instruction per route; house-interleaved order',
                    arm_order='Native first, six heads rotated by registered group id; inherited worker',
                    primary='Mean over three seeds of paired CONCAT minus NATIVE SR on NEW routes only',
                    secondary='CONCAT minus LOCAL by seed; SPL, action cost, rescues/regressions and house results',
                    interpretation='Positive descriptive replication if mean delta >0 and at least two seeds >0; report all seeds and uncertainty, no automatic adoption',
                    no_score_selection=True, prior_public_benchmark_exposure=True, new_houses=False, full_1839=False,
                    exposure_scope='Route-disjoint from all registered StreamVLN branch manifests. Other historical Qwen experiments exposed this public split; not blind testing.',
                    gpu_indices=list(range(8)), chunk_pairs=8, gpu_session_hours_limit=64,
                    wall_hours_limit=24, artifact_limit_gib=32, automatic_score_retries=0,
                    mechanism=dict(dataset='All64 cached DEV teacher/preservation traces', models='All six final heads',
                                   controls=['FULL', 'SHAM', 'RECENT8_RAW', 'RECENT8_NORM', 'REVERSED_OLD_NORM', 'CURRENT_NORM'],
                                   match='Per-query full-state L2 norm; same actor features, native logits and weights',
                                   scope='Readout diagnostics on recorded trajectories, not autonomous SR or proof of semantic state',
                                   hyperparameter_selection=False, known_queries_only=True))
    for record in protocol['heads'].values():
        assert u.sha(record['path']) == record['sha256']
    u.write(run/'PROTOCOL.json', protocol)
    u.write(run/'DATA_MANIFEST.json', dict(episodes=entries, source=str(source), source_sha256=u.sha(source)))
    u.write(run/'SPLIT_AUDIT.json', dict(source_episodes=1839, selected_routes=len(eligible), selected_instructions=len(entries),
            route_overlap=0, excluded_houses=sorted(excluded_houses), excluded_route_keys=sorted(excluded),
            manifest_evidence=evidence, outcome_fields_not_used=True, selection=selection,
            limitation=protocol['exposure_scope']))
    u.write(run/'STATUS.json', dict(status='PREPARED', phase='CPU_MECHANISM', planned_groups=len(entries),
                                  planned_executions=len(entries)*7, gpu_hours=0))
    freeze(run)
    print(json.dumps(dict(run=str(run), routes=len(entries), executions=len(entries)*7)))


if __name__ == '__main__':
    main()
