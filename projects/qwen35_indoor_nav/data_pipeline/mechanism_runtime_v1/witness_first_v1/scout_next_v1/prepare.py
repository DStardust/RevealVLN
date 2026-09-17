"""CPU-only, deterministic next-six-FIT-house preparation. Never launches workers."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if p.name == 'vla')
LINE = ROOT / 'projects/qwen35_indoor_nav'
PREVIOUS = HERE.parent / 'scout_v1'
MANIFEST = LINE / 'data_pipeline/mechanism_scale_v1/acceptance_v1/candidates.json'
SPLIT = LINE / 'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
SOURCE = ROOT / 'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
EXCLUDED = ('17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r')
GPUS = ((1, 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'),
        (2, 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'))


def sha(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT), str(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, 'FROZEN_OUTPUT_MISMATCH: ' + str(path)
        return
    with path.open('x') as f:
        f.write(text)


def choose(rows, fit, excluded=EXCLUDED, limit=6):
    """Manifest first-encounter order; no ranking on scout outcomes."""
    chosen, seen = [], set()
    for row in rows:
        house = row['house_id']
        if house in seen:
            continue
        seen.add(house)
        assert house in fit and row['split'] == 'FIT'
        if house in excluded:
            continue
        assert row['house_candidate_rank'] == 0
        chosen.append(copy.deepcopy(row))
        if len(chosen) == limit:
            return chosen
    raise ValueError('INSUFFICIENT_NEW_FIT_HOUSES')


def source_positions(episodes, house):
    selected = [e for e in episodes if Path(e['scene_id']).parent.name == house]
    values = {tuple(p) for e in selected for p in [e['start_position']] + e['reference_path']}
    assert values and selected, 'NO_REGISTERED_SOURCE_POSITIONS: ' + house
    return [list(p) for p in sorted(values)], len(selected)


def shard_configs(base, candidates):
    assert len(candidates) == 6 and len({r['house_id'] for r in candidates}) == 6
    configs = []
    for i, (gpu, uuid) in enumerate(GPUS):
        cfg = copy.deepcopy(base)
        cfg.update(node='Q35N_WITNESS_SCOUT_NEXT_V1_SHARD_' + str(i),
            runtime_allowed=False, executable=False, training_allowed=False,
            gpu_device=gpu, gpu_uuid=uuid, candidates=candidates[i*3:(i+1)*3],
            shard_id=i, intended_output_root=str((HERE / ('shard_' + str(i)) / 'run_v1').relative_to(ROOT)),
            runtime_adapter_ready=False, main_agent_runtime_approval_required=True,
            immutable_algorithm_source=str(PREVIOUS.relative_to(ROOT)),
            metadata_permission_roles_are_not_chosen_witness_roles=True,
            household_selection='manifest_first_encounter_excluding_original_three_no_outcome_ranking')
        cfg.pop('source_configuration_sha256', None)
        cfg.update(source_sha256=sha(SOURCE), manifest_sha256=sha(MANIFEST), split_sha256=sha(SPLIT))
        configs.append(cfg)
    return configs


def main():
    assert sha(SOURCE) == 'f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
    rows = json.loads(MANIFEST.read_text())
    fit = set(json.loads(SPLIT.read_text())['FIT'])
    assert len({r['house_id'] for r in rows}) == 43
    selected = choose(rows, fit)
    with gzip.open(SOURCE, 'rt') as f:
        episodes = json.load(f)['episodes']
    inventory = []
    assets = []
    for row in selected:
        house = row['house_id']
        row['source_positions'], episode_count = source_positions(episodes, house)
        row['assets'] = {}
        for suffix in ('.glb', '.house', '.navmesh', '_semantic.ply'):
            path = ROOT / f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            row['assets'][str(path)] = sha(path)
            assets.append(path)
        row['scene_glb'] = next(p for p in row['assets'] if p.endswith('.glb'))
        # Keep the already-frozen permission metadata only. Actual witness roles are
        # discovered anew from complete backend.objects and real Compiler events.
        inventory.append(dict(house_id=house, split='FIT', source_episodes=episode_count,
            source_positions=len(row['source_positions']), assets=len(row['assets']),
            metadata_role_purpose='backend_permission_only_not_witness_certification'))
    configs = shard_configs(json.loads((PREVIOUS / 'PREPARED_CONFIG.json').read_text()), selected)
    configs_paths = []
    for i, cfg in enumerate(configs):
        folder = HERE / ('shard_' + str(i))
        folder.mkdir(exist_ok=True)
        p = folder / 'PREPARED_CONFIG.json'
        save(p, cfg)
        configs_paths.append(p)
    inventory_path = HERE / 'SOURCE_INVENTORY.json'
    save(inventory_path, dict(selected=inventory, excluded_houses=list(EXCLUDED),
        manifest_house_count=43, selected_house_count=6, executable=False,
        new_physical_families=0, new_training_families=0,
        combined_independent_budget=dict(total_seconds=8400, supervision_seconds_per_shard=4500,
            total_actions=120000, total_disk_bytes=16*1024**3,
            total_content_bytes=12*1024**3, ram_bytes_per_shard=8*1024**3)))
    previous_lock = json.loads((PREVIOUS / 'INPUT_LOCK.json').read_text())
    paths = list(HERE.glob('*.py')) + [HERE / 'README.md', MANIFEST, SPLIT, SOURCE,
        PREVIOUS / 'PREPARED_CONFIG.json', PREVIOUS / 'INPUT_LOCK.json', inventory_path] + configs_paths + assets
    # Lock actual algorithm and dependencies; do not copy or hash the active output tree.
    paths += [ROOT / p for p in previous_lock if p.endswith('.py') or p.endswith('/SHA256SUMS')]
    lock = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(HERE / 'INPUT_LOCK.json', lock)
    result = dict(status='CPU_PREPARED_NOT_EXECUTABLE', houses=inventory, shards=2,
        source_and_code_assets_locked=len(lock), gpu_operations=0,
        simulator_imports=0, simulations=0, new_families=0,
        runtime_adapter_required=True, scientific_pass=False)
    save(HERE / 'PREPARATION_RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
