import ast
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]


def read(p):return json.loads(p.read_text())
def write(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    audit=read(OUT/'AUDIT_RESULT.json');gen=read(OUT/'GENERATION_RESULT.json')
    assert audit['integrity_pass'] and gen['generation_yield_pass']
    phases=[OUT,OUT/'recovery_v1']
    for p in phases:
        assert read(p/'LEASE_RESTORED.json')['restored']
        assert read(p/'EXECUTION_RESULT.json')['cleanup_complete']
    protected=[]
    for name in ['Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1','Q35N_P2R1_SPEC_CORRECTIONS_V1',
                 'Q35N_G0R_DEPENDENCY_RECOVERY_V1','Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2']:
        r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=LINE/'reviews'/name,capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
        protected.append({'node':name,'verified_hashes':r.stdout.count(': OK'),'pass':True})
    write('PROTECTED_CHECKS.json',protected)
    for p in OUT.rglob('*.py'):ast.parse(p.read_text())
    samples=[s for p in phases for s in read(p/'RESOURCE_SAMPLES.json')]
    lower0=read(OUT/'COUNTS_LIVE.json');counts1=read(OUT/'recovery_v1/ACTUAL_COUNTS.json')
    resource={'gpu_phase_wall_seconds':sum(read(p/'EXECUTION_RESULT.json')['wall_seconds'] for p in phases),
        'max_worker_rss_kib':max(s.get('child_rss_kib',0) for s in samples),
        'max_device_total_memory_mib':max(s['memory_mib'] for s in samples),
        'primitive_actions_exact_total':None,
        'primitive_actions_saved_lower_bound':lower0['primitive_actions']+counts1['primitive_actions'],
        'reason_total_unknown':'First worker SIGTERM interrupted final counting during uncommitted route_016; its in-flight work is not zero',
        'files_bytes_before_sealing':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        'placeholder_release_cycles':2,'placeholder_restore_cycles':2,'real_tasks_stopped':0,
        'new_model_loads':0,'new_training_runs':0,'new_learned_policy_eval_episodes':0}
    write('RESOURCE_RESULT.json',resource)
    write('result.json',{'decision':'ORDINARY_DATA_PILOT_PASS','ordinary_data_pipeline_pass':True,
        'attempted_unique_routes':100,'certified_routes':audit['certified_routes'],'houses':5,
        'instruction_records':audit['instruction_records'],'unique_rgb_frames':audit['unique_png_frames'],
        'unique_route_decisions':audit['unique_route_decisions'],
        'instruction_conditioned_decisions':audit['instruction_conditioned_supervised_decisions'],
        'failure_counts':audit['failure_counts'],'recovery_retained':True,
        'index':'TRAINING_INDEX.jsonl','index_sha256':sha(OUT/'TRAINING_INDEX.jsonl'),
        'mechanism_certified_families':0,'scientific_pass':False,'navigation_gain':None,
        'training_started':False,'gpu_cleanup_and_placeholder_restore':True,
        'next_execution_approved':False})
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and 'cache' not in p.parts]
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(OUT)}\n')
    r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    print(f'{len(files)} sealed delivery files verified. Ordinary data pilot PASS; no method scientific PASS.')
    print(json.dumps(resource,indent=2))


if __name__=='__main__':main()
