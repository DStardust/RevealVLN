"""Freeze two exposed families for a physical interface test, never an effect test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LINE, ROOT, RESEARCH, RUNTIME, c


def main():
    source = LINE/'research/continuation_memory_v1/multifamily_v7/DATA.json'
    prior = c.read(source)
    rows = []
    houses = set()
    excluded = []
    for item in prior['audit']['families']:
        if item['house'] in houses:
            continue
        export = LINE/item['export']
        manifest = c.read(export/'MANIFEST.json')
        candidate = manifest['candidate']
        maximum = max(map(len,candidate['histories'].values()))+max(map(len,candidate['continuations'].values()))
        if maximum > 500:
            excluded.append(dict(family_id=manifest['family_id'], max_complete_decisions=maximum))
            continue
        inventory = c.read(export.parent/'SEMANTIC_INVENTORY.json')['objects']
        roles = {}
        for role, pair in manifest['compiler_config']['roles'].items():
            ids = manifest['compiler_config']['eligible'][role]
            raws = {inventory[str(i)]['raw'].lower().replace('#', ' ').strip() for i in ids}
            assert len(raws) == 1, 'ROLE_RAW_NAMES_REQUIRE_EXPLICIT_MAPPING'
            roles[role] = dict(mpcat40=pair[0], room=pair[1], raw_match=dict(mode='exact', value=raws.pop()))
        assets = candidate['context']['asset_config']
        assert all(Path(path).resolve().is_relative_to(ROOT) for path in assets)
        scene = next(path for path in assets if path.endswith('.glb'))
        rows.append(dict(family_id=manifest['family_id'], house=item['house'],
            export=item['export'], manifest_sha256=c.sha(export/'MANIFEST.json'),
            candidate=candidate, compiler=manifest['compiler_config'], roles=roles,
            scene=scene, assets=assets, original_training_admission=manifest['training_admission']))
        houses.add(item['house'])
        if len(rows) == 2:
            break
    assert len(rows) == 2
    protocol = dict(id='Q35N_V15_RAW_PHYSICAL_INTERFACE_R2', gpu=1,
        gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',
        stage='Only test raw physical execution without numerical joins; no model training or effect claim.',
        families=rows, data_source_sha256=c.sha(source), max_decisions=20000,
        excluded_by_full500_budget=excluded,
        revision='R1 stopped after one complete trace when a later source suffix exceeded 500. R2 checks all combinations before simulation; no score selection or larger decision budget.',
        previous_protocol_sha256=c.sha(RESEARCH/'RAW_PROTOCOL_R1.json'),
        max_episode_decisions=500, environment_seed=0, output_gib=20,
        expected_traces=sum(len(x['candidate']['histories'])*len(x['candidate']['continuations']) for x in rows),
        interior_state_assignment=False, original_assets_read_only=True,
        acceptance='Actual raw RGB window, action window, and full pose equality measured separately. No snapping, reconstruction, copied RGB, or approximate matching to manufacture equality.',
        next_stage='After interface repair, generate legal families with event-free detour and terminal-only controls, freeze two data sizes and B1/B2/Ours V14 architecture before training. Independently held family/house evaluation.',
        source_hashes={str(path.relative_to(LINE)):c.sha(path) for path in [
            *HERE.glob('*.py'), RUNTIME/'habitat_backend.py', RUNTIME/'guard.py',
            RUNTIME/'feedback_generation_v1/store.py', LINE/'data_pipeline/mechanism_factory_v2/compiler.py']})
    c.write(RESEARCH/'RAW_PROTOCOL_R2.json', protocol, True)
    print(dict(families=[x['family_id'] for x in rows], expected_traces=protocol['expected_traces']))


if __name__ == '__main__':
    main()
