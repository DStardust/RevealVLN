import ast
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def main():
    r=read(OUT/'result.json');assert r['interface_pass'] and not r['scientific_pass']
    assert read(OUT/'EXECUTION_RESULT.json')['worker_cleanup'] and read(OUT/'LEASE_RESTORED.json')['restored']
    assert read(OUT/'ENVIRONMENT_ACCEPTANCE.json')['installed_modeling_matches_tag_source']
    for name,h in read(OUT/'PROBE_CODE_LOCK.json').items():assert sha(OUT/name)==h,name
    for p in OUT.glob('*.py'):ast.parse(p.read_text())
    protected=[]
    for rel in ['data_pipeline/ordinary_pilot_v1','reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1',
                'reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2','reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1']:
        c=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=LINE/rel,capture_output=True,text=True)
        assert c.returncode==0,c.stdout+c.stderr
        protected.append({'directory':rel,'verified_hashes':c.stdout.count(': OK'),'pass':True})
    save('PROTECTED_CHECKS.json',protected)
    sample=read(OUT/'RESOURCE_SAMPLES.json')
    sizes={}
    for rel in ['.envs/q35n_qwen_g2_v1','.cache/q35n_qwen_g2_v1','runtime/models/Qwen3.5-2B_15852e8']:
        sizes[rel]=int(subprocess.check_output(['du','-sb',str(LINE/rel)],text=True).split()[0])
    assert sum(sizes.values())<40*1024**3
    save('RESOURCE_RESULT.json',{'artifact_sizes_bytes':sizes,'sum_bytes':sum(sizes.values()),
        'max_worker_rss_kib':max(s['rss_kib'] for s in sample),
        'max_device_memory_mib':max(s['memory_mib'] for s in sample),
        'peak_torch_allocated_bytes':r['peak_cuda_allocated_bytes'],
        'measured_network_transport_bytes':None,'model_asset_bytes':sum(x['bytes'] for x in read(OUT/'MODEL_FILE_LOCK.json'))+sum(x['bytes'] for x in read(OUT/'MODEL_SUPPLEMENT_LOCK.json')),
        'new_training_runs':0,'diagnostic_optimizer_updates':1,'new_model_loads':1,
        'new_learned_navigation_episodes':0,'real_tasks_stopped':0,'placeholder_restored':True})
    # The initial source helper fetched modeling, then got a documented 404 on processing.
    save('SOURCE_FETCH_PARTIAL_RECORD.json',{'modeling_source_sha256':sha(OUT/'official_source/modeling_qwen3_5.py'),
        'missing_source':'processing_qwen3_5.py','http_status':404,
        'resolution':'Model officially uses Qwen3VLProcessor; installed code and official config verified. No remote source code executed.'})
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts]
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(OUT)}\n')
    c=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT,capture_output=True,text=True)
    assert c.returncode==0,c.stdout+c.stderr
    print(f'{len(files)} sealed G2 files verified; minimal interface PASS, scientific_pass=false')


if __name__=='__main__':main()
