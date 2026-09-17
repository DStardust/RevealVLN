"""Validate actual diagnostic content and seal the closed P0 node."""
import collections
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare import sha, sealed, LINE
from loader import npy_pixels


def main():
    assert not (HERE/'SHA256SUMS').exists()
    r = json.loads((HERE/'result.json').read_text())
    state = json.loads((LINE/'STATUS.json').read_text())
    assert state['multifamily_p0_confirmed_motion_actions'] == r['confirmed_physical_actions'] == 2393
    assert state['multifamily_new_certified_families'] == r['new_geometric_families'] == 0
    assert not state['multifamily_runtime_executable'] and not state['multifamily_p1_allowed']
    assert r['gpu_cleanup_complete']
    counts = collections.Counter()
    for path in sorted((HERE/'run/content').glob('*.npy')):
        expected, kind, _ = path.name.split('.')
        npy_pixels(path.read_bytes(), expected, kind)
        counts[kind] += 1
    json_count = 0
    for path in HERE.rglob('*.json'):
        if 'cache' not in path.relative_to(HERE).parts:
            json.loads(path.read_text())
            json_count += 1
    for path in HERE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target:
                assert (path.parent/target.split('#')[0]).exists(), target
    samples = json.loads((HERE/'run/RESOURCE_SAMPLES.json').read_text())
    validation = {'status_result_consistency': True, 'JSON_files_parsed': json_count,
                  'all_content_arrays_validated': dict(counts),
                  'array_counts_include_partial_probe_content_not_training_samples': True,
                  'local_report_links_valid': True,
                  'peak_sampled_RAM_bytes': max(s['ram_bytes'] for s in samples),
                  'peak_sampled_GPU_mib': max(s['gpu']['memory_mib'] for s in samples),
                  'framework_check': 'PASS_bash_scripts_research_sh_check',
                  'scientific_pass': False}
    with (HERE/'FINAL_VALIDATION.json').open('x') as f:
        json.dump(validation, f, indent=2)
    files = [p for p in HERE.rglob('*') if p.is_file() and p != HERE/'SHA256SUMS'
             and 'cache' not in p.relative_to(HERE).parts and '__pycache__' not in p.relative_to(HERE).parts]
    assert all(p.resolve().is_relative_to(HERE) for p in files)
    with (HERE/'SHA256SUMS').open('x') as f:
        for path in sorted(files):
            f.write(sha(path)+'  '+str(path.relative_to(HERE))+'\n')
    sealed(HERE)
    sealed(HERE.parent)
    print(json.dumps({'sealed_files': len(files), **validation}, indent=2))


if __name__ == '__main__':
    main()
