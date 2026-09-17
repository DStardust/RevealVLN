"""CPU-only actual online transport, training pixel alignment and baseline regression."""
import ast,base64,hashlib,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
REVIEW=LINE/'reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10'
CASE=LINE/'closed_loop_bench/ordinary_action_aligned_history_dev_v10'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_initialized()
    h=runpy.run_path(str(HERE/'history_r1.py'));oldh=runpy.run_path(str(HERE/'history.py'))
    checks=[]
    assert (HERE/'history_r1.py').read_text()==(HERE/'history.py').read_text().replace('[-(STRIDE+1)];','[-(STRIDE+1):];')
    for t in range(501):
        expected=[0] if t==0 else [max(0,t-8),t]
        assert h['indices'](t)==oldh['indices'](t)==expected
        assert h['choose'](list(range(501)),t)==expected
    for bad in (-1,1.5,True):
        try:h['indices'](bad);raise RuntimeError('ACCEPTED_BAD_INDEX')
        except AssertionError:pass
    checks.append('same_data_selector_all_501_indices_and_negative_cases')
    common=runpy.run_path(str(CASE/'common.py'))
    w=common['Window']();blobs=[bytes([i%256])*(224*224*3) for i in range(121)]
    actions=[]
    for t,blob in enumerate(blobs):
        payload=dict(done=False,rgb=base64.b64encode(blob).decode())
        if t==0:payload['instruction']='Walk ahead and turn left.'
        action=None if t==0 else common['ACTIONS'][(t-1)%3]
        assert w.receive(payload,action)
        if action is not None:actions.append(action)
        selected=h['choose'](blobs,t)
        assert w.images==selected and len(w.frames)<=9 and w.executed==actions[-8:]
        audit=w.input_audit()
        assert audit['input_frame_indices']==h['indices'](t)
        assert audit['input_rgb_sha256']==[hashlib.sha256(b).hexdigest() for b in selected]
        item=w.item();assert [im.tobytes() for im in item['images']]==selected
        assert set(item)=={'instruction','images','executed'}
    assert not w.receive(dict(done=True),'STOP')
    assert w.receive(dict(done=False,rgb=base64.b64encode(blobs[0]).decode(),instruction='New route'))
    assert w.decision_t==0 and w.executed==[] and w.images==[blobs[0]] and len(w.frames)==1
    for payload,action in [(dict(done=True,goal=[0,0,0]),'STOP'),
        (dict(done=False,rgb=base64.b64encode(blobs[0]).decode(),pose=[0,0,0]),'turn_left'),
        (dict(done=False,rgb=base64.b64encode(blobs[0]).decode()),'STOP')]:
        try:w.receive(payload,action);raise RuntimeError('ACCEPTED_PRIVILEGED_OR_BAD_ACTION')
        except ValueError:pass
    checks.append('actual_online_window_121_frames_pixels_actions_terminal_reset_and_whitelist')
    d=runpy.run_path(str(HERE/'data.py'));rows,report=d['load_rows']();store=d['SampleStore'](rows)
    verified=0
    for idx in (0,10,100,1000,8699,9000,17000,20000,37000):
        record=d['adapter']['OrdinaryRecord'](rows[idx])
        for t in sorted({0,1,min(8,len(record)-1),min(12,len(record)-1)}):
            item=store.get(idx,t);meta=record.decision_metadata(t)
            refs=h['choose'](record._refs,t)
            expected=[Path(x).stem for x in refs]
            assert [hashlib.sha256(im.tobytes()).hexdigest() for im in item['images']]==expected
            assert item['target']==d['ACTIONS'].index(meta['supervision']['target_action'])
            assert item['executed']==meta['policy']['executed_actions']
            assert all(im.mode=='RGB' and im.size==(224,224) for im in item['images'])
            verified+=1
    for idx in range(37114,37146):
        item=store.get(idx,0);view=store.recovery[rows[idx]['advice_record_id']]
        assert [hashlib.sha256(im.tobytes()).hexdigest() for im in item['images']]==view['rgb_sha256']
        assert item['target']==rows[idx]['target'] and item['executed']==view['executed_actions']
        assert set(item)=={'instruction','images','executed','target'}
        verified+=1
    checks.append('actual_training_ordinary_and_recovery_pixel_hash_target_history')
    audit=read(HERE/'VIEW_AUDIT.json');assert audit['status']=='PASS' and audit['planned_view_target_conflicts']==0
    for p,digest in audit['sources'].items():assert sha(Path(p))==digest,p
    assert sha(HERE/'RECOVERY_VIEW.json')==audit['view_sha256']
    assert sha(HERE/'RECOVERY_PROVENANCE.json')==audit['recovery_provenance_sha256']
    assert audit['planned_occurrences']==99047 and audit['ordinary_occurrences']==89172 and audit['recovery_occurrences']==9875
    checks.append('frozen_full_plan_view_audit_unchanged_99047_no_conflicting_targets')
    prior=runpy.run_path(str(LINE/'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1/analyze.py'))
    new=runpy.run_path(str(REVIEW/'analyze.py'));base=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
    assert prior['case_report'](base)==new['case_report'](base)
    checks.append('complete_100_baseline_analysis_exact_regression')
    for folder,names in [(HERE,('model.py','train_filestore.py','control.py','supervise_filestore.py','launcher.py','lease_run.py')),
                          (CASE,('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'))]:
        reuse=runpy.run_path(str(folder/'reuse.py'))
        for name in names:ast.parse(reuse['source'](name))
    for folder in (HERE,REVIEW,CASE):
        for p in folder.glob('*.py'):ast.parse(p.read_text())
    launch=runpy.run_path(str(CASE/'launch.py'))
    assert launch['OUT']==CASE/'run_001' and callable(launch['_scan'].tree_size)
    assert not (CASE/'run_001').exists() and not torch.cuda.is_initialized()
    checks.append('all_runtime_sources_compile_actual_launcher_import_no_gpu')
    result=dict(status='PASS',unix=time.time(),checks=checks,checks_count=len(checks),actual_training_samples=verified,
      simulated_transport_frames=121,history_version='history_r1.py',history_sha256=sha(HERE/'history_r1.py'),
      source_history_preserved=True,training_updates=0,simulator_actions=0,navigation_gain_verified=False)
    with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
