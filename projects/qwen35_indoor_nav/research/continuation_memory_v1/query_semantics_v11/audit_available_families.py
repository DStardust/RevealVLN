"""Inventory and verify every existing SEE2 export without new simulation/training."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    began = time.monotonic()
    assets = LINE/'data_pipeline/mechanism_runtime_v1/witness_first_v1'
    paths = sorted(assets.glob('batch_execution_v1/*/run_v1/bundles/*/export_v4/MANIFEST.json'))
    paths += sorted(assets.glob('diversity*/run_v1/bundles/*/export_v4/MANIFEST.json'))
    original = read(HERE.parent/'multifamily_v7/DATA.json')
    held = {h for h, split in original['audit']['house_split'].items() if split=='check'}
    write(HERE/'AVAILABLE_FAMILIES_SCOPE.json', dict(
        sources={str(p.relative_to(LINE)): sha(p) for p in paths},
        source_code_sha256=sha(Path(__file__)), original_check_houses=sorted(held),
        selection='All existing exports at scope creation; not selected by model scores.',
        scope='CPU export integrity and semantic coverage only; no training admission or novel data claim.'))
    loader_module = load('pool_loader', LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
    core = load('pool_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    rows, failures, duplicates, fingerprints = [], [], [], {}
    for number, path in enumerate(paths, 1):
        manifest = read(path)
        source = str(path.parent.relative_to(LINE))
        try:
            compiler = core.Compiler(**manifest['compiler_config'])
            loader = loader_module.FamilyLoader(path.parent, compiler)
            labels = loader.validate_supervision_contract()
            candidate = manifest['candidate']
            # IDs/provenance are not independent trajectories. Compare executable content.
            physical = {k: candidate[k] for k in ('position','context_key','yaw_bin',
                         'public_tail','histories','continuations','numerical_join')}
            content = dict(physical=physical, compiler=manifest['compiler_config'])
            key = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            house = manifest['group']['house_id']
            prior = fingerprints.setdefault(key, source)
            duplicate = prior != source
            if duplicate:
                duplicates.append(dict(source=source, same_as=prior, executable_content_sha256=key))
            rows.append(dict(source=source, family_id=manifest['family_id'], house=house,
                split='check' if house in held else 'fit', duplicate=duplicate,
                manifest_sha256=sha(path), executable_content_sha256=key,
                labels=labels, roles=manifest['compiler_config']['roles'],
                tasks=manifest['compiler_config']['tasks'],
                original_training_admission=manifest['training_admission'],
                original_scientific_pass=manifest['scientific_pass'],
                runtime_certification=manifest['runtime_certification'],
                numerical_join=manifest['candidate']['numerical_join'],
                prefix_decisions=manifest['prefix_decisions'], action_owners=manifest['ce_unique_owners']))
        except Exception as exc:
            failures.append(dict(source=source, error=repr(exc)))
        print('AUDITED', number, '/', len(paths), 'VALID', len(rows), 'ERRORS', len(failures), flush=True)
    fit_roles = set()
    for row in rows:
        if row['split']=='fit' and not row['duplicate']:
            for task in row['tasks'].values():
                fit_roles.update(tuple(row['roles'][task[key]]) for key in ('anchor','terminal'))
    check_roles = set()
    for row in rows:
        if row['split']=='check' and not row['duplicate']:
            for task in row['tasks'].values():
                check_roles.update(tuple(row['roles'][task[key]]) for key in ('anchor','terminal'))
    write(HERE/'AVAILABLE_FAMILIES_AUDIT.json', dict(
        status='EXISTING_EXPORT_CPU_AUDIT_COMPLETE', seconds=time.monotonic()-began,
        scope_sha256=sha(HERE/'AVAILABLE_FAMILIES_SCOPE.json'),
        total_exports=len(paths), verified_exports=len(rows), failures=failures,
        exact_executable_duplicates=duplicates,
        unique_exports=len(rows)-len(duplicates),
        unique_house_counts=dict(Counter(r['house'] for r in rows if not r['duplicate'])),
        unique_split_counts=dict(Counter(r['split'] for r in rows if not r['duplicate'])),
        fit_task_roles=sorted(fit_roles), check_task_roles=sorted(check_roles),
        check_roles_missing_from_fit=sorted(check_roles-fit_roles), rows=rows,
        original_check_houses_retained=True, original_training_admission=False,
        new_physical_replays=0, new_encoder_forwards=0, new_optimizer_updates=0,
        limitation='Exporter integrity is not physical certification. Numerical joins, missing controls and raw certificates retain their original status. Content deduplication does not make remaining shared routes independent. No research gain is asserted.'))


if __name__ == '__main__':
    main()
