"""Seal completed G1F discovery evidence; does not run simulator or change gates."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]

if __name__=='__main__':
    assert not (OUT/'SHA256SUMS').exists()
    result=json.loads((OUT/'result.json').read_text())
    assert result['candidate_configurations_completed']==256
    assert not result['data_contract_pass'] and not result['scientific_pass']
    assert result['frozen_families']==0 and result['exported_training_labels']==0
    for p in OUT.rglob('*.json'):json.loads(p.read_text())
    for p in OUT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    lock=json.loads((OUT/'discovery_pruned_CODE_LOCK.json').read_text())
    for name in ('engine.py','run_phase.py'):
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==lock[name]
    fp=json.loads((OUT.parent/'Q35N_G0R_DEPENDENCY_RECOVERY_V1/ENVIRONMENT_FINGERPRINT.json').read_text())
    env=LINE/'.envs/q35n_habitat_v017_g0r'
    p=env/'lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==fp['sysconfig_sha256']
    p=LINE/'runtime/q35n_habitat_v017_g0r/src/habitat-sim/build/CMakeCache.txt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==fp['cmake_cache_sha256']
    for phase in ('preview','discovery','discovery_pruned'):
        assert json.loads((OUT/f'{phase}_EXECUTION.json').read_text())['cleanup_complete']
    status=json.loads((LINE/'STATUS.json').read_text())
    assert not status['g1_execution_approved'] and not status['training_allowed']
    (OUT/'FINAL_CHECKS.json').write_text(json.dumps({'json_and_python_syntax':True,'pruned_source_lock':True,
        'runtime_sysconfig_and_build_fingerprint_unchanged':True,'gpu_cleanup_records_pass':True,
        'scientific_claim_guards':True,'node_closed':True},indent=2)+'\n')
    records=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            assert not p.is_symlink()
            records.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n')
    (OUT/'SHA256SUMS').write_text(''.join(records))
    c=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT,capture_output=True,text=True)
    assert c.returncode==0,c.stdout+c.stderr
    print(f'{len(records)}/{len(records)} delivery hashes verified; discovery closed without scientific PASS.')
