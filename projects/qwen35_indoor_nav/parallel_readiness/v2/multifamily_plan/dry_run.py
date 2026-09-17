"""CPU manifest-only planning. Deliberately contains no simulator execution path."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(manifest, split, protocol):
    if protocol.get('executable') is not False:
        raise ValueError('CPU planner requires executable=false')
    houses = split['fit_pilot']
    reserved = split['reserved_unassigned']
    if len(houses) != 5 or len(reserved) != 56 or set(houses) & set(reserved):
        raise ValueError('frozen 5/56 split violated')
    routes = defaultdict(dict)
    records = Counter()
    for row in manifest:
        house = row['scene_id']
        if house not in set(houses + reserved):
            raise ValueError('source house outside frozen train inventory')
        if row['source_split'] != 'train' or row['source_sha256'] != protocol['source']['raw_source_sha256']:
            raise ValueError('source lineage mismatch')
        records[house] += 1
        # Intentionally do not retain/export source instructions or coordinates.
        key = row['physical_source_route_sha256']
        item = routes[house].setdefault(key, {'instruction_aliases': 0, 'reference_waypoints': row['reference_waypoints']})
        item['instruction_aliases'] += 1
    inventory = [{'house_id': h, 'frozen_group': 'FIT_PILOT' if h in houses else 'RESERVED_UNASSIGNED',
                  'source_records': records[h], 'source_physical_routes': len(routes[h]),
                  'old_exposure': split['legacy_status'][h], 'new_asset_reads': 0,
                  'semantic_eligibility': 'unknown', 'reachability': 'unknown'} for h in houses + reserved]
    candidates = []
    for rank in protocol['enumeration']['route_ranks']:
        for house in houses:
            keys = sorted(routes[house])
            if rank >= len(keys):
                raise ValueError('not enough routes; no silent replacement')
            key = keys[rank]
            candidates.append({'candidate_id': 'MP5_%02d' % len(candidates), 'stage': 'P0' if rank == 0 else 'P1_PROPOSED',
                               'house_id': house, 'source_physical_route_sha256': key, 'route_rank': rank,
                               'source_metadata': routes[house][key], 'frozen_house_group': 'FIT_PILOT',
                               'runtime_u_positions': None, 'semantic_eligibility': 'unknown', 'reachability': 'unknown',
                               'new_geometry': 'unknown', 'certified': False, 'production_executable': False})
    return inventory, candidates


def protected_paths():
    names = ['data_pipeline/v1/SOURCE_INVENTORY.json', 'data_pipeline/v1/SOURCE_CANDIDATES.jsonl',
             'data_pipeline/ordinary_pilot_v1/SPLIT_FREEZE.json',
             'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/FAMILY_MANIFEST.json',
             'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/SHA256SUMS',
             'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/SUPERVISION_ONLY.jsonl',
             'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/COMPILER_CODE_LOCK.json']
    lock = json.loads((LINE / names[-1]).read_text())
    names += list(lock)
    return sorted(set(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', required=True)
    args = parser.parse_args()
    assert args.dry_run
    protocol = json.loads((OUT / 'protocol.json').read_text())
    before = {n: digest(LINE / n) for n in protected_paths()}
    split = json.loads((LINE / protocol['source']['split']).read_text())
    rows = [json.loads(line) for line in (LINE / protocol['source']['manifest']).read_text().splitlines()]
    inventory, candidates = build(rows, split, protocol)
    dump(OUT / 'HOUSE_INVENTORY.json', inventory)
    dump(OUT / 'CANDIDATES.json', candidates)
    after = {n: digest(LINE / n) for n in protected_paths()}
    if before != after:
        raise RuntimeError('protected source content changed during CPU planning')
    dump(OUT / 'PROTECTED_HASH_AUDIT.json', {'before': before, 'after': after, 'unchanged': True,
         'scope': 'listed metadata and sealed compiler sources; not a full rehash of image arrays'})
    dump(OUT / 'CPU_RESULT.json', {'decision': 'CPU_DRY_RUN_COMPLETE_RUNTIME_NOT_AUTHORIZED',
         'source_records': sum(x['source_records'] for x in inventory),
         'source_physical_routes': sum(x['source_physical_routes'] for x in inventory),
         'houses': len(inventory), 'candidate_bundles': len(candidates), 'first_batch_candidates': 5,
         'new_asset_reads': 0, 'new_scene_observations': 0, 'new_certified_families': 0,
         'existing_family_not_recounted': True, 'protected_files_unchanged': True,
         'production_executable': False, 'scientific_pass': False})
    print('CPU dry-run complete: 20 seed candidates, 0 new certified families, no runtime access.')


if __name__ == '__main__':
    main()
