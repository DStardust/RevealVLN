"""Register actual public assets and the existing exposed development denominator."""
import argparse
import gzip
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as u


def main(run):
    run.mkdir(parents=True, exist_ok=False)
    origin = u.read(u.ORIGIN / 'ASSET_VERIFIED_MANIFEST.json')
    config = u.read(u.MODEL / 'config.json')
    assert origin[0]['repo'].endswith('_v1_3')
    source = u.PROJECT / 'sft_acceptance/ordinary_branch_decision_v21/runs/pilot_002/DATA_MANIFEST.json'
    rows = u.read(source)['evaluation']
    assert len(rows) == 100 and len({x['episode']['episode_id'] for x in rows}) == 100
    # Upstream instruction vocabulary is required by the Habitat dataset loader.
    fixture = u.read(u.ORIGIN / 'FIXTURE_MANIFEST.json')[0]
    source_json = next(k for k in fixture['files'] if k.endswith('.json.gz'))
    import json
    vocab = json.loads(gzip.decompress((u.ASSETS / source_json).read_bytes()))['instruction_vocab']
    template = (u.CODE / 'config/vln_r2r.yaml').read_text()
    fixtures = run / 'fixtures'; fixtures.mkdir()
    selections = []
    for row in rows:
        ep = row['episode']; index = row['id']
        data = fixtures / f'{index:03d}.json.gz'
        data.write_bytes(gzip.compress(json.dumps(dict(episodes=[ep], instruction_vocab=vocab)).encode(), mtime=0))
        config_path = fixtures / f'{index:03d}.yaml'
        text = template.replace('scenes_dir: data/scene_datasets/', f'scenes_dir: {u.ASSETS}/third_party/ETP-R1/data/scene_datasets/')
        text = text.replace('data_path: data/datasets/r2r/{split}/{split}.json.gz', f'data_path: {data}')
        text = text.replace('split: val_seen', 'split: val_unseen').replace('habitat:\n', 'habitat:\n  seed: 42\n', 1)
        config_path.write_text(text)
        scene = u.ASSETS / 'third_party/ETP-R1/data/scene_datasets/mp3d' / row['house'] / (row['house'] + '.glb')
        assert scene.is_file(), str(scene)
        selections.append(dict(id=index, episode_id=ep['episode_id'], house=row['house'], config=str(config_path),
            instruction=ep['instruction']['instruction_text'], original_source_sha256=u.sha(source)))
    protocol = dict(id='Q35N_STRONG_BACKBONE_RECOVERY_V1', status='EXPLORATORY_NOT_A_PAPER_RESULT',
        backbone=origin[0]['repo'], revision=origin[0]['revision'], seed=42, model=str(u.MODEL), python=str(u.PYTHON),
        source_commit=subprocess.check_output(['git', '-C', str(u.CODE), 'rev-parse', 'HEAD'], text=True).strip(),
        evaluation='Exposed INTERNAL_DEV100; native public StreamVLN sensor/action/history protocol',
        sensor=dict(rgb=[640,480], hfov=79, processor=[384,384]), max_steps=500, num_frames=32, num_history=8,
        num_future_steps=4, dtype='bfloat16', attention='flash_attention_2', hidden_size=config['hidden_size'],
        compatibility='Project-owned Python3.10 / Torch2.8 CUDA12.8 / Habitat0.2.4; not authors hardware bitwise reproduction',
        imported_environment_read_only=True, old_records_read_only=True, identity_pairs=2, native_episodes=100,
        gpu_indices=[2,3,4,5,6,7], gpu_session_hours_limit=12, wall_hours_limit=4,
        policy='Unmodified upstream action generation/parser; no native STOP override or outcome-score replacement',
        first_method='Task-history differences train actual action preferences; exact-state supervision is the matched B2',
        method_status='IMPLEMENTATION_PENDING_REAL_DATA_ADMISSION',
        method_claim='UNTESTED; no memory-method gain is inferred from stronger public backbone scores',
        no_score_based_retries=True, no_automatic_full_unseen=True)
    u.write(run / 'PROTOCOL.json', protocol)
    u.write(run / 'DATA_MANIFEST.json', dict(episodes=selections, source=str(source), source_sha256=u.sha(source),
        selection='Unchanged V21 INTERNAL_DEV100 list; no selection using StreamVLN outcomes', original_results_reused=False))
    files = list(u.HERE.glob('*.py')) + list(fixtures.iterdir()) + [run / 'PROTOCOL.json', run / 'DATA_MANIFEST.json']
    tracked = subprocess.check_output(['git','-C',str(u.CODE),'ls-files'], text=True).splitlines()
    files += [u.CODE / p for p in tracked if p.endswith(('.py','.yaml','.json'))]
    u.write(run / 'SOURCE_LOCK.json', dict(files={str(p):u.sha(p) for p in files}, upstream_commit=protocol['source_commit']))
    u.write(run / 'ASSET_PROVENANCE.json', dict(origin=str(u.ORIGIN), origin_manifest_sha256=u.sha(u.ORIGIN / 'ASSET_VERIFIED_MANIFEST.json'),
        files=origin, simulator_origin=str(u.ORIGIN / 'SIMULATOR_ASSET_SNAPSHOT.json'),
        existing_admission_is_historical=True, old_model_and_environment_will_not_be_modified=True))
    u.write(run / 'STATUS.json', dict(status='PREPARED', phase='ASSET_VERIFY', completed_native=0, planned_native=100,
        method_training_started=False, method_benefit='UNKNOWN'))
    (u.HERE / 'LAST_RUN.txt').write_text(run.name + '\n')
    print(str(run))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    main(u.HERE / 'runs' / parser.parse_args().run_id)
