"""Resolve only the preselected P0 route seeds; does not authorize rendering."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
sys.path.insert(0, str(RUNTIME))
from prepare import LINE, ROOT, sha, sealed, asset_config
from core_bridge import planning


def physical_hash(episode):
    physical = {k: episode[k] for k in ('scene_id', 'start_position', 'start_rotation', 'reference_path', 'goals')}
    return hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def resolve(records, episodes):
    selected = planning.select_candidates(records)
    wanted = {r['source_physical_route_sha256']: r for r in selected}
    found = {}
    aliases = {}
    for ep in episodes:
        # No image/asset access and no instruction retention for unselected houses.
        house = Path(ep['scene_id']).parent.name
        if house not in {r['house_id'] for r in selected}:
            continue
        key = physical_hash(ep)
        if key not in wanted:
            continue
        assert house == wanted[key]['house_id']
        configs = planning.enumerate_configurations(ep['start_position'], ep['reference_path'])
        assert key not in found or found[key] == configs, 'INCONSISTENT_ROUTE_ALIAS'
        found[key] = configs
        aliases[key] = aliases.get(key, 0)+1
    assert set(found) == set(wanted), 'MISSING_REGISTERED_ROUTE'
    return [{**r, 'configurations': found[r['source_physical_route_sha256']],
             'resolved_instruction_alias_count': aliases[r['source_physical_route_sha256']]}
            for r in selected]


def prepare(out):
    assert out.resolve().is_relative_to(RUNTIME) and not out.exists()
    plan = LINE/'parallel_readiness/v2/multifamily_plan'
    sealed(plan)
    protocol = json.loads((plan/'protocol.json').read_text())
    source = (ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz').resolve()
    assert source.is_relative_to(ROOT) and sha(source) == protocol['source']['raw_source_sha256']
    with gzip.open(source, 'rt') as f:
        episodes = json.load(f)['episodes']
    candidates = resolve(json.loads((plan/'CANDIDATES.json').read_text()), episodes)
    old = json.loads((RUNTIME/'smoke_v1/EXECUTION_CONFIG.json').read_text())
    roles = old['roles']
    names = {'anchor_A': ('客厅', '电视'), 'anchor_B': ('厨房', '水槽')}
    tasks = {task: {'anchor': role, 'terminal': 'terminal',
                   'instruction': f'先连续两帧看见{names[role][0]}内的{names[role][1]}，再连续两帧看见卧室内的床，然后立即停止。'}
             for task, role in [('task_A', 'anchor_A'), ('task_B', 'anchor_B')]}
    for row in candidates:
        row['assets'] = asset_config(row['house_id'])
        row['scene_glb'] = next(p for p in row['assets'] if p.endswith('.glb'))
    cfg = {'phase': 'five_fixed_bundle_constructability_P0', 'runtime_allowed': False,
           'gpu_device': 2, 'roles': roles, 'tasks': tasks, 'candidates': candidates,
           'budget': dict(planning.DEFAULT_LIMITS), 'disk_cap_bytes': 16*1024**3,
           'ram_cap_bytes': 16*1024**3, 'gpu_cap_mib': 8192,
           'source_sha256': sha(source), 'plan_sha256': sha(plan/'SHA256SUMS'),
           'smoke_result_sha256': None, 'stop_after_batch': True,
           'training_admission': False, 'scientific_pass': False}
    out.mkdir()
    with (out/'CONFIG_DRAFT.json').open('x') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = prepare(args.output.resolve())
    print(json.dumps({'runtime_allowed': False, 'candidates': len(cfg['candidates']),
                      'configurations': [len(r['configurations']) for r in cfg['candidates']]}))
