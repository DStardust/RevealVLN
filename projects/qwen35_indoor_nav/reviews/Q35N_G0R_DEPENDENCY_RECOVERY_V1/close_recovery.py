"""Collect engineering evidence without upgrading scientific claims."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ENV = LINE / '.envs/q35n_habitat_v017_g0r'
SRC = LINE / 'runtime/q35n_habitat_v017_g0r/src/habitat-sim'

def save(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def main():
    assert not (OUT/'result.json').exists(), 'Closed results are immutable; use a new node for another run'
    checks = {}
    for name in ('Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1',
                 'Q35N_P2R1_SPEC_CORRECTIONS_V1',
                 'Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1'):
        p = subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT.parent/name,
                           capture_output=True,text=True)
        checks[name] = {'pass':p.returncode == 0,'ok_count':p.stdout.count(': OK'),
                        'stdout':p.stdout,'stderr':p.stderr}
    assert all(c['pass'] for c in checks.values())
    save('PROTECTED_ARTIFACT_CHECK.json', checks)
    for p in OUT.glob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
    for p in OUT.glob('*.json'):
        json.loads(p.read_text())
    paths = {'environment':ENV,'runtime_including_quarantine':LINE/'runtime/q35n_habitat_v017_g0r',
             'cache':LINE/'.cache/q35n_habitat_v017_g0r','recovery_artifacts':OUT,
             'original_g0r_artifacts':OUT.parent/'Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1'}
    sizes = {name:int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for name,p in paths.items()}
    assert sum(sizes.values()) < 40*1024**3
    assert sizes['environment'] < 12*1024**3
    assert sizes['runtime_including_quarantine'] < 8*1024**3
    inventory = json.loads((OUT/'WHEEL_INVENTORY.json').read_text())
    lock = ''.join(f'{r["name"]}=={r["version"]} --hash=sha256:{r["sha256"]}\n' for r in inventory)
    (OUT/'requirements.lock.txt').write_text(lock)
    origins = json.loads((OUT/'OFFICIAL_WHEEL_ORIGINS.json').read_text())
    rss = {}
    for log in OUT.glob('*.log'):
        m = re.search(r'Maximum resident set size \(kbytes\): (\d+)',log.read_text())
        if m:
            rss[log.name] = int(m.group(1))
    steps = {p.name:json.loads(p.read_text()) for p in OUT.glob('*.result.json')}
    steps['smoke_v1.result.json'] = steps.pop('smoke.result.json')
    steps['smoke.result.json'] = json.loads((OUT/'smoke_v2/smoke.result.json').read_text())
    success = all(steps.get(name+'.result.json',{}).get('returncode') == 0 for name in
                  ('download','install_local','configure_relocated','package_existing','install_native','check_native','smoke'))
    smoke = steps.get('smoke.result.json',{})
    success = success and smoke.get('renderer_pass',False)
    save('RESOURCE_MEASUREMENTS.json',{
        'directory_bytes':sizes,'total_bytes':sum(sizes.values()),
        'verified_downloaded_wheel_bytes':sum(r['bytes'] for r in inventory),
        'official_verification_metadata_bytes':sum(r['metadata_response_bytes'] for r in origins),
        'network_control_and_tls_bytes_exact':None,
        'network_accounting_note':'Wheel payload and verification JSON measured; pip index/metadata/TLS and original failed control traffic not exhaustively metered. No weights or scene downloads.',
        'time_max_child_rss_kib':rss,'compile_parallel_jobs':4,
        'aggregate_host_peak_rss_exact':None,
        'ram_measurement_note':'time maximum is per-child, not sum of simultaneous compiler RSS; do not mislabel it as exact host peak.',
        'gpu':smoke})
    installed = subprocess.check_output([str(ENV/'bin/python3'),'-I','-m','pip','--isolated','list','--format=json'],text=True)
    sysconfig = ENV/'lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py'
    save('ENVIRONMENT_FINGERPRINT.json',{'prefix':str(ENV),'python':'3.10.20',
        'installed_packages':json.loads(installed), 'habitat_commit':'856d4b08c1a2632626bf0d205bf46471a99502b7',
        'headless':True,'cuda_kernels':False,'bullet':False,
        'sysconfig_relocation_modified':True,'sysconfig_sha256':hashlib.sha256(sysconfig.read_bytes()).hexdigest(),
        'cmake_cache_sha256':hashlib.sha256((SRC/'build/CMakeCache.txt').read_bytes()).hexdigest(),
        'historical_bad_prefix_build_quarantined_not_accepted':True,
        'strict_original_upstream_environment_reproduction':False})
    counts_path = OUT/'smoke_v2/SMOKE_COUNTS.json'
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else None
    save('result.json',{
        'node':'Q35N_G0R_DEPENDENCY_RECOVERY_V1',
        'closed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'decision':'RUNTIME_AND_RENDERER_ENGINEERING_PASS' if success else 'ENGINEERING_INCOMPLETE',
        'dependency_acquisition_pass':True,'official_wheel_hashes_verified':len(origins),
        'runtime_pass':success,'renderer_pass':smoke.get('renderer_pass',False),
        'accepted_smoke_directory':'smoke_v2',
        'scientific_pass':False,'navigation_gain':None,'new_training_runs':0,'model_loads':0,
        'family_replays':0,'new_navigation_evaluation_episodes':0,'actual_smoke_counts':counts,
        'historical_failures_preserved':True,'steps':steps,
        'next_scientific_dependency':'G1F physical family and label acceptance; not automatically authorized',
        'resource_measurement_limitations':['No exhaustive network-control byte meter','No exact aggregate concurrent host RSS peak'],
        'g1_execution_approved_by_this_script':False})
    print(json.dumps({'runtime_pass':success,'renderer_pass':smoke.get('renderer_pass',False),
                      'disk_gib':sum(sizes.values())/1024**3,'counts':counts}))

if __name__ == '__main__':
    main()
