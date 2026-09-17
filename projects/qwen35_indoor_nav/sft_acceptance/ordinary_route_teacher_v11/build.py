"""CPU-only exact common-input relabel/filter plan and metadata-only FP32 bridge."""
import collections,copy,hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_v6'
BASE=HERE.parent/'ordinary_expanded_v1'
DATA=LINE/'data_pipeline/ordinary_route_teacher_v11/run_001'
PREVDATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
BRIDGE=HERE.parent/'ordinary_onpolicy_fp32_master_v8/initial_fp32_master_from_best4000.pt'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(n,x):
    with (HERE/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'BUILD_AUDIT.json').exists()
    assert read(DATA/'RESULT.json')['data_gate'] and read(DATA/'RESULT.json')['ordered_route_teacher']
    assert read(DATA/'PIXEL_AUDIT.json')['passed']
    for folder in (DATA,PREVDATA):
        for p,d in read(folder/'DATA_SEAL.json')['files'].items():assert sha(p)==d,p
    oldp=read(OLD/'PROTOCOL_FILESTORE.json')
    assert sha(OLD/'TRAINING_ROWS.json')==oldp['snapshot']['training_index_sha256']
    assert sha(OLD/'SAMPLE_INDEX.jsonl')==oldp['sample_index_sha256']
    assert sha(OLD/'PLAN.json')==oldp['sampling_plan_sha256']
    rec=read(OLD/'SOURCE_RECONCILIATION.json')
    assert sha(OLD/'SOURCE_RECONCILIATION.json')=='bb705217386cde6385ff733e622d622c41e46c2c121639517ebbb662b0ebdac9'
    for p,d in rec['old_sources_sha256'].items():assert sha(p)==d,p
    oldrows=read(OLD/'TRAINING_ROWS.json');old_ids={x['advice_record_id'] for x in oldrows[37114:]}
    assert old_ids==set(rec['eligible_new_advice']) and len(old_ids)==4983
    before={x['record_id']:x for x in lines(PREVDATA/'POLICY_INPUTS.jsonl')}
    after={x['record_id']:x for x in lines(DATA/'POLICY_INPUTS.jsonl')}
    targets={x['record_id']:x for x in read(DATA/'TRAINING_TARGETS.json')}
    assert set(after)==set(targets)
    eligible=sorted(old_ids&set(targets));houses={h for k in eligible for h in targets[k]['scene_groups']}
    assert len(eligible)>=1000 and len(houses)>=8
    assert not set(eligible)&set(rec['old_new_input_overlap'])
    for k in eligible:
        assert after[k]==before[k] and set(after[k])=={'record_id','instruction','rgb_sha256','executed_actions'}
        assert targets[k]['target'] in range(4)
    save('SOURCE_RECONCILIATION.json',dict(status='PASS',eligible_common_inputs=eligible,
      dropped_old_input_ids=sorted(old_ids-set(eligible)),ignored_new_input_ids=sorted(set(targets)-old_ids),
      inherited_old_all_overlap_exclusion_sha256=sha(OLD/'SOURCE_RECONCILIATION.json'),
      original_inputs_byte_equivalent=True,no_old_ordinary_labels_changed=True,
      new_teacher_gate_sha256=sha(DATA/'RESULT.json'),no_new_independent_routes=True))
    oldplan=read(OLD/'PLAN.json');wanted={i for rank in oldplan for batch in rank for i in batch}
    sampled={};tail={}
    with (OLD/'SAMPLE_INDEX.jsonl').open() as f:
        for i,line in enumerate(f):
            if i in wanted or i>=2650347:
                v=json.loads(line)
                if i in wanted:sampled[i]=v
                if i>=2650347:tail[i]=v
    assert len(tail)==4983 and set(sampled)==wanted
    by_id={oldrows[v[0]]['advice_record_id']:(i,v) for i,v in tail.items()}
    mapping={};new_entries=[];rows=copy.deepcopy(oldrows[:37114]);changes=collections.Counter();changed_weights=0
    A=['move_forward','turn_left','turn_right','STOP']
    for key in eligible:
        oldi,v=by_id[key];target=targets[key]['target'];item=after[key];hist=item['executed_actions']
        weight=3.2 if not hist or A[target]!=hist[-1] else 1.0
        newidx=2650347+len(new_entries);mapping[oldi]=newidx
        new_entries.append([len(rows),0,target,weight,v[4]])
        row=copy.deepcopy(oldrows[v[0]])
        row.update(target=target,source='AUDITED_ONPOLICY_ROUTE_TEACHER',
                   quality_tier='ACTUAL_VISITED_STATE_ORDERED_ROUTE_ONE_STEP_CHECKED')
        assert row['scene_groups']==targets[key]['scene_groups']
        rows.append(row);changes[A[v[2]]+'->'+A[target]]+=1;changed_weights+=int(weight!=v[3])
    save('TRAINING_ROWS.json',rows)
    index=HERE/'SAMPLE_INDEX.jsonl';prefix=hashlib.sha256()
    with index.open('xb') as dst,(BASE/'SAMPLE_INDEX.jsonl').open('rb') as src:
        for block in iter(lambda:src.read(2**20),b''):dst.write(block);prefix.update(block)
        for v in new_entries:dst.write((json.dumps(v)+'\n').encode())
    assert prefix.hexdigest()==read(BASE/'PROTOCOL_FILESTORE.json')['sample_index_sha256']
    plan=[[[i if i<2650347 else mapping[i] for i in batch if i<2650347 or i in mapping] for batch in rank] for rank in oldplan]
    assert len(plan)==3 and all(len(rank)==1000 and all(batch for batch in rank) for rank in plan)
    ordinary=0;recover=0;draw_changes=collections.Counter();maxbatch=0;weighted_dropped=0
    for rank,newrank in zip(oldplan,plan):
        for oldbatch,newbatch in zip(rank,newrank):
            assert [i for i in oldbatch if i<2650347]==[i for i in newbatch if i<2650347]
            assert sum(sampled[i][4] for i in oldbatch if i<2650347 or i in mapping)<=6144
            for i in oldbatch:
                if i<2650347:
                    ordinary+=1;assert oldrows[sampled[i][0]]['source']=='R2R'
                elif i in mapping:
                    recover+=1;draw_changes[A[sampled[i][2]]+'->'+A[new_entries[mapping[i]-2650347][2]]]+=1
                else:weighted_dropped+=sampled[i][3]
    counts=[sum(map(len,rank)) for rank in plan];planned=sum(counts)
    maxbatch=max(sum(len(plan[r][j]) for r in range(3)) for j in range(1000))
    assert ordinary==89172 and planned==ordinary+recover and .08<recover/planned<.12
    assert 99047-planned==9875-recover
    save('PLAN.json',plan)
    import torch
    assert not torch.cuda.is_initialized()
    assert sha(BRIDGE)=='aa4e3d3bb9be97e4b906bc56834c3cc008c17eeef52ed51c24ac0bbb052466e5'
    state=torch.load(BRIDGE,map_location='cpu',weights_only=True)
    original=copy.deepcopy(state)
    assert state['cursor']==dict(epoch=0,position=0,updates=0,decisions=0)
    assert state['global_decisions']==state['charged_compute_decisions']==0
    assert len(state['trainable'])==len(state['optimizer']['state'])==28
    assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in state['trainable'].values())
    for x in state['optimizer']['state'].values():
        assert int(x['step'])==4000 and x['exp_avg'].dtype==x['exp_avg_sq'].dtype==torch.float32
    state['binding']=dict(sample_index_sha256=sha(index),source_checkpoint_sha256=sha(BRIDGE),
                          stage='ROUTE_TEACHER_NEW_INDEX_ONLY_SAME_FP32_VALUES_MOMENTS_RNG')
    initial=HERE/'initial_from_best4000_fp32_metadata_rebind.pt'
    with initial.open('xb') as f:torch.save(state,f)
    back=torch.load(initial,map_location='cpu',weights_only=True)
    def exact(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b) and a.dtype==b.dtype
        elif isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:exact(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert type(a)==type(b) and len(a)==len(b)
            for x,y in zip(a,b):exact(x,y)
        else:assert a==b
    for k in original:
        if k!='binding':exact(original[k],back[k])
    exact(state,back)
    save(initial.name+'.json',dict(sha256=sha(initial),cursor=back['cursor'],global_decisions=0,charged_compute_decisions=0,
      source_fp32_bridge_sha256=sha(BRIDGE),metadata_only_change=True,optimizer_reset=False,optimizer_step_offset=4000,unix=time.time()))
    audit=dict(status='PASS',unix=time.time(),recovery_unique_inputs=len(eligible),recovery_unique_drawn=len({i for rank in plan for batch in rank for i in batch if i>=2650347}),
      planned_updates=1000,planned_decisions=planned,rank_planned_decisions=counts,ordinary_draws=ordinary,recovery_draws=recover,
      removed_unavailable_recovery_draws=9875-recover,removed_recovery_weight_sum=weighted_dropped,
      recovery_decision_fraction=recover/planned,changed_inflection_weights_unique=changed_weights,
      unique_target_transitions=dict(changes),draw_target_transitions=dict(draw_changes),houses=sorted(houses),
      old_ordinary_order_labels_weights_exact=True,old_relative_recovery_order_preserved=True,encoded_input_content_unchanged=True,
      initial_parameter_reload_exact=True,optimizer_moments_rng_preserved=True,new_index_metadata_rebind_only=True,
      initial_path=str(initial),initial_sha256=sha(initial),initial_receipt_sha256=sha(str(initial)+'.json'),
      old_fp32_bridge_sha256=sha(BRIDGE),same_decision_count_claim=False,max_global_batch_decisions=maxbatch,
      all_data_FIT=True,automatic_training_restart=False,training_updates=0,simulator_actions=0)
    save('BUILD_AUDIT.json',audit)
    assert not torch.cuda.is_initialized();print(json.dumps(audit))
if __name__=='__main__':main()
