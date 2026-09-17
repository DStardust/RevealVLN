"""Read-only third pass over completed navigation artifacts; no model or simulator."""
import collections,hashlib,json,math,statistics,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_policy_preservation_dev_v9'
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
TRAIN=LINE/'sft_acceptance/ordinary_policy_preservation_v9'
REVIEW=LINE/'reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def source_check():
    locks=[CASE/'SOURCE_LOCK.json',REVIEW/'SEAL.json']
    files={}
    for p in locks:
        for name,digest in read(p)['files'].items():
            if name in files:assert files[name]==digest,name
            files[name]=digest
    for name,digest in files.items():assert sha(name)==digest,name
    return dict(files=len(files),bytes=sum(Path(x).stat().st_size for x in files),lock_hashes={str(p):sha(p) for p in locks})
def shared_check():
    a=read(BASE/'PROTOCOL.json');b=read(CASE/'PROTOCOL.json')
    fixed=['seed','environment_seed','episode_count','house_count','houses','house_counts','max_steps','rgb_size','hfov',
        'camera_height','agent_height','agent_radius','forward_m','turn_deg','allow_sliding','success_distance',
        'greedy','optimizer_updates','metric_backend','ndtw_backend','lanes','gt_path','comparison_batch_size_forced']
    for k in fixed:assert a[k]==b[k],k
    for n in ['EPISODES_PRIVILEGED.json','PARITY_FIXTURES.json','GEOMETRY_PREFLIGHT.json']:assert sha(BASE/n)==sha(CASE/n),n
    assert sha(CASE/'EPISODES_PRIVILEGED.json')=='2f41a84a575aabe3ddbb5d9d41e29bd9c779eff93b673b80d4c06db0ba953b16'
    tp=read(TRAIN/'PROTOCOL_FILESTORE.json');r=read(TRAIN/'formal/attempt_001/RESULT.json')
    assert r['status']=='EPOCHS_COMPLETED' and r['stop']==[]
    assert r['cursor']==tp['expected_final_cursor']
    assert r['global_decisions']==r['charged_compute_decisions']==99047
    assert r['teacher_forward_decisions']==99047 and r['total_policy_forward_decisions']==198094
    assert tp['reference_kl_lambda']==1.0 and tp['reference_temperature']==1.0
    assert tp['reference_checkpoint']['sha256']=='aa4e3d3bb9be97e4b906bc56834c3cc008c17eeef52ed51c24ac0bbb052466e5'
    assert sha(tp['reference_checkpoint']['path'])==tp['reference_checkpoint']['sha256']
    assert b['training_protocol_sha256']==sha(TRAIN/'PROTOCOL_FILESTORE.json')
    assert b['sample_index_sha256']==tp['sample_index_sha256']
    assert sha(b['checkpoint'])==b['checkpoint_sha256']==read(REVIEW/'FINAL_CHECKPOINT_ACCEPTANCE.json')['checkpoint_sha256']
    assert read(CASE/'run_001/MODEL_LOADED.json')['checkpoint_sha256']==b['checkpoint_sha256']
    lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
    assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None and lease['external_processes_stopped']==0
    return dict(matched_protocol_keys=fixed,checkpoint_sha256=b['checkpoint_sha256'],training_stage_updates=1000,
        optimizer_step=5000,training_decisions=99047,teacher_forward_decisions=99047,total_policy_forward_decisions=198094,holders_restored=True)
def replay_official_metrics(x,episode):
    import runpy
    import numpy as np
    from types import SimpleNamespace as NS
    module=runpy.run_path(str(LINE/'closed_loop_bench/r2r_ce_tiny_v1/metrics.py'))
    class SavedTrace:
        def __init__(self):self.index=0
        def get_agent_state(self):return NS(position=np.asarray(x['positions'][self.index],dtype=np.float32))
        def geodesic_distance(self,*args):return x['distances'][self.index]
    sim=SavedTrace()
    metrics=module['OfficialMetrics'](sim,episode['goals'][0]['position'])
    assert len(x['positions'])==len(x['distances'])==x['steps']+1
    for k in range(1,x['steps']+1):
        sim.index=k;v=metrics.update(bool(x['stopped'] and k==x['steps']))
    close(v['spl'],x['spl']);close(v['success'],x['success'])
    close(v['distance_to_goal'],x['navigation_error_m'])
    return v
def aggregate(case):
    r=read(case/'run_001/RESULT.json')
    assert r['status']=='COMPLETE' and r['completed']==100 and r['trace_audit_passed']
    rows=sorted([read(p) for p in (case/'run_001/lanes').glob('lane_*/episode_*.json')],key=lambda e:e['index'])
    assert len(rows)==100 and [x['index'] for x in rows]==list(range(100))
    inputs=read(case/'EPISODES_PRIVILEGED.json')
    for x,i in zip(rows,inputs):
        assert x['episode_id']==i['episode_id'] and x['trajectory_id']==i['trajectory_id']
        assert x['instruction']==i['instruction']['instruction_text']
        assert x['house']==i['scene_id'].split('/')[1]
        assert x['steps']<=500 and not x['service_failure']
        assert int(x['success'])==int(x['stopped'] and x['navigation_error_m']<3)
        replay_official_metrics(x,i)
        close(x['sdtw'],x['success']*x['ndtw'])
    metrics={k:statistics.mean(e[v] for e in rows) for k,v in dict(sr='success',spl='spl',ndtw='ndtw',osr='oracle_success').items()}
    for k,v in metrics.items():close(v,r[k])
    actions=sum(x['steps'] for x in rows)
    assert actions==r['environment_actions']==r['audited_actions']
    assert dict(collections.Counter(x['failure_category'] for x in rows))==r['failure_categories']
    return rows,metrics,actions
def final_check():
    a,am,aa=aggregate(BASE);b,bm,ba=aggregate(CASE)
    assert [(x['index'],x['episode_id'],x['house']) for x in a]==[(x['index'],x['episode_id'],x['house']) for x in b]
    delta={k:bm[k]-am[k] for k in ['sr','spl','ndtw']}
    wins=sum(not x['success'] and y['success'] for x,y in zip(a,b))
    losses=sum(x['success'] and not y['success'] for x,y in zip(a,b))
    positive=delta['sr']>0 and delta['spl']>=0 and delta['ndtw']>=-.01
    r=read(REVIEW/'RESULT.json');w=read(REVIEW/'WORKFLOW_RESULT.json')
    assert r['positive_development_signal']==w['positive_development_signal']==positive
    assert r['paired']['wins']==wins and r['paired']['losses']==losses
    launch=read(CASE/'run_001/LAUNCH_RESULT.json')
    assert launch['status']=='COMPLETE' and launch['returncode']==0 and not launch['cleanup']['remaining']
    assert launch['foreign_processes_signaled']==[] and launch['borrowed_holders']==[] and launch['optimizer_updates']==0
    assert launch['gpus_after'][1]['contexts']==[]
    with urllib.request.urlopen('http://127.0.0.1:18766/api/status',timeout=15) as f:monitor=json.load(f)
    assert monitor['monitor_version']=='ordinary_policy_preservation_v9'
    assert monitor['onpolicy_training']['positive_navigation_result']==positive
    return dict(before=am,after=bm,delta=delta,wins=wins,losses=losses,audited_actions=ba,
        positive_development_signal=positive,independent_confirmation=False,
        evaluation_owned_processes_cleaned=True,original_monitor_synchronized=True)
if __name__=='__main__':
    assert sys.argv[1:] in (['pre'],['final'],['test'])
    stage=sys.argv[1]
    if stage=='test':
        a=aggregate(BASE);b=aggregate(CASE.parent/'ordinary_onpolicy_adapt_dev_v6r3')
        assert a[1]['sr']==.21 and b[1]['sr']==.19 and a[2]==19122 and b[2]==20853
        result=dict(status='PASS',tested_complete_episodes=200,official_metrics_replayed=True,
            simulator_actions=0,recorded_distance_not_new_geodesic_query=True,
            threshold_unchanged=True,source_sha256=sha(__file__))
        with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result));sys.exit(0)
    out=HERE/(stage.upper()+'_RESULT.json');assert not out.exists()
    result=dict(status='PASS',stage=stage,unix=time.time(),scope='Repeatedly exposed fixed100 INTERNAL_DEV; engineering only',
        shared=shared_check(),sources=source_check(),official_metric_cpu_replay=True,recorded_geodesic_distances_not_recomputed=True,extra_training_updates=0,extra_simulator_actions=0,code_sha256=sha(__file__))
    if stage=='final':result['navigation']=final_check()
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))
