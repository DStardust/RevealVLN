"""CPU real loader, preserved plan positions, exact pixel targets and eval transport."""
import ast,hashlib,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_route_teacher_dev_v11'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_initialized()
    d=runpy.run_path(str(HERE/'data.py'));rows,report=d['load_rows']();binding=read(HERE/'DATA_BINDING.json')
    assert report['training_index_sha256']==binding['snapshot']['training_index_sha256']
    samples=d['load_sample_index'](HERE/'SAMPLE_INDEX.jsonl',binding['sample_index_sha256'],binding['snapshot']['decisions_per_epoch'])
    plan=d['plan_epoch_batches'](samples,6144,1209,0,3);audit=read(HERE/'BUILD_AUDIT.json')
    old=read(HERE.parent/'ordinary_onpolicy_adapt_v6/PLAN.json')
    assert [sum(map(len,r)) for r in plan]==audit['rank_planned_decisions']
    assert sum(map(len,[b for r in plan for b in r]))==98979
    assert audit['ordinary_draws']==89172 and audit['recovery_draws']==9807 and audit['removed_unavailable_recovery_draws']==68
    checks=[]
    for a,b in zip(old,plan):
        assert len(a)==len(b)==1000
        for oa,nb in zip(a,b):
            assert [i for i in oa if i<2650347]==[i for i in nb if i<2650347]
            assert nb and sum(samples[i]['est'] for i in nb)<=6144
    checks.append('all_3000_rank_batches_keep_ordinary_order_nonempty_token_budget')
    src=HERE.parent/'ordinary_expanded_v1/SAMPLE_INDEX.jsonl';h=hashlib.sha256();remaining=src.stat().st_size
    with (HERE/'SAMPLE_INDEX.jsonl').open('rb') as f:
        while remaining:
            block=f.read(min(2**20,remaining));assert block;h.update(block);remaining-=len(block)
    assert h.hexdigest()==sha(src)
    checks.append('ordinary_index_prefix_bytes_exact')
    store=d['SampleStore'](rows)
    counts=0
    targets={x['record_id']:x for x in read(LINE/'data_pipeline/ordinary_route_teacher_v11/run_001/TRAINING_TARGETS.json')}
    chosen=list(range(2650347,2650347+100))+[rank[0][0] for rank in plan]
    for i in chosen:
        sample=samples[i];item=store.get(sample['record_idx'],sample['t'])
        assert set(item)=={'instruction','images','executed','target'} and item['target']==sample['target']
        assert all(im.mode=='RGB' and im.size==(224,224) for im in item['images'])
        assert len(item['images'])<=2 and len(item['executed'])<=8
        if i>=2650347:
            key=rows[sample['record_idx']]['advice_record_id'];value=store.inputs[key]
            assert targets[key]['target']==item['target']
            assert [hashlib.sha256(im.tobytes()).hexdigest() for im in item['images']]==value['rgb_sha256']
            expected=3.2 if not item['executed'] or d['ACTIONS'][item['target']]!=item['executed'][-1] else 1.
            assert sample['weight']==expected
        counts+=1
    checks.append('103_actual_training_items_pixels_targets_same_inflection_rule')
    for folder,names in [(HERE,('train_filestore.py','model.py','control.py','supervise_filestore.py','launcher.py','lease_run.py')),
                         (CASE,('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'))]:
        r=runpy.run_path(str(folder/'reuse.py'))
        for n in names:ast.parse(r['source'](n))
    common=runpy.run_path(str(CASE/'common.py'))
    oldcommon=runpy.run_path(str(CASE.parent/'r2r_ce_tiny_v1/common.py'))
    import base64
    w=common['Window']()
    for t in range(12):
        p=dict(done=False,rgb=base64.b64encode(bytes([t])*(224*224*3)).decode())
        if t==0:p['instruction']='Walk to the room.'
        assert w.receive(p,None if t==0 else 'move_forward')
        assert [im[0] for im in w.images]==([0] if t==0 else [t-1,t])
        assert len(w.executed)==min(t,8)
    checks.append('actual_deployment_unchanged_last2_RGB_last8_actions')
    launch=runpy.run_path(str(CASE/'launch.py'),run_name='CPU_IMPORT_ONLY')
    assert launch['OUT']==CASE/'run_001' and callable(launch['_scan'].tree_size)
    assert not (CASE/'run_001').exists() and not torch.cuda.is_initialized()
    result=dict(status='PASS',unix=time.time(),checks=checks,actual_training_items=counts,training_updates=0,gpu_actions=0,
       planned_decisions=audit['planned_decisions'],rank_decisions=audit['rank_planned_decisions'])
    with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
