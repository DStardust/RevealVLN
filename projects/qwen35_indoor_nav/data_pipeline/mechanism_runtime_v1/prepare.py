"""Freeze physical inputs without loading a renderer or touching SFT."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
DATA = LINE/'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def sealed(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split(None, 1)
        path = (root/relative).resolve()
        assert path.is_relative_to(root.resolve()) and sha(path) == expected, relative


def asset_config(house):
    assert house.isalnum()
    directory = ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d'/house
    config = {}
    for suffix in ('.glb', '.house', '.navmesh', '_semantic.ply'):
        path = (directory/(house+suffix)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), str(path)
        config[str(path)] = sha(path)
    return config


def smoke(output, gpu):
    assert output.resolve().is_relative_to(HERE) and output != HERE
    sealed(DATA)
    sealed(LINE/'data_pipeline/mechanism_factory_v2')
    old = json.loads((LINE/'reviews/Q35N_G1R_TASK_INSTANCE_V3/FROZEN_CANDIDATE.json').read_text())
    assets = asset_config('17DRP5sb8fy')
    registered = json.loads((DATA/'SOURCE_AND_CONFIG.json').read_text())['source']['asset_hashes']
    assert {'sha256:'+x for x in assets.values()} == set(registered)
    pair = {'D': 'anchor_A', 'K': 'anchor_B', 'B': 'terminal', 'L': 'irrelevant'}
    inventory = json.loads((DATA/'TASK_ROLE_INVENTORY.json').read_text())
    roles = {pair[k]: {'mpcat40': v[0], 'room': v[1], 'raw_match': {
        'mode': 'token' if v[0] == 'chair' else 'exact',
        'value': {'chair': 'chair', 'tv_monitor': 'tv', 'bed': 'bed', 'sink': 'sink'}[v[0]]}}
             for k, v in inventory['kinds'].items()}
    tasks = {'task_A': {'anchor': 'anchor_A', 'terminal': 'terminal', 'instruction': old['task_configuration']['tasks']['g_T_v3']},
             'task_B': {'anchor': 'anchor_B', 'terminal': 'terminal', 'instruction': old['task_configuration']['tasks']['g_K_v3']}}
    config = {'phase': 'existing_family_backend_smoke', 'runtime_allowed': False,
        'house_id': '17DRP5sb8fy', 'scene_glb': next(p for p in assets if p.endswith('.glb')),
        'assets': assets, 'gpu_device': gpu, 'roles': roles, 'tasks': tasks,
        'expected_eligible': {pair[k]: v for k, v in inventory['eligible'].items()},
        'candidate': {'position': old['u']['position'], 'yaw_bin': old['yaw_bin'], 'public_tail': old['public_tail'],
            'histories': { {'H_T': 'H_A', 'H_K': 'H_B', 'H_T_I': 'H_A_I'}[k]: v for k, v in old['histories'].items()},
            'continuations': { {'C0': 'C0', 'C_T': 'C_A', 'C_K': 'C_B'}[k]: v for k, v in old['continuations'].items()},
            'numerical_join': old['numerical_join']},
        'budget': {'total_actions': 12000, 'total_seconds': 600, 'discovery_actions': 12000,
                   'discovery_seconds': 600, 'certification_actions': 12000, 'certification_seconds': 600},
        'disk_cap_bytes': 2*1024**3, 'ram_cap_bytes': 16*1024**3, 'gpu_cap_mib': 8192,
        'seed': 1109, 'expected_real_replays': 9, 'scientific_pass': False,
        'source_manifest_sha256': sha(DATA/'SHA256SUMS'), 'source_exposure': 'interface_only_old_line_exposed'}
    output.mkdir(exist_ok=False)
    with (output/'CONFIG_DRAFT.json').open('x') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps({'phase': smoke(args.output, args.gpu)['phase'], 'runtime_allowed': False}))
