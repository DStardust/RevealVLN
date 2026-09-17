"""Construct an immutable input VIEW, not new labels, routes, images, or draws."""
import collections,concurrent.futures,hashlib,json,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_v6'
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001'
h=runpy.run_path(str(HERE/'history.py'))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    z=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):z.update(b)
    return z.hexdigest()
def save(name,x):
    with (HERE/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'VIEW_AUDIT.json').exists()
    old=read(OLD/'PROTOCOL_FILESTORE.json')
    assert sha(OLD/'TRAINING_ROWS.json')==old['snapshot']['training_index_sha256']
    assert sha(OLD/'SAMPLE_INDEX.jsonl')==old['sample_index_sha256']
    assert sha(OLD/'PLAN.json')==old['sampling_plan_sha256']
    for p,d in read(DATA/'DATA_SEAL.json')['files'].items():assert sha(p)==d,p
    rr=read(OLD/'TRAINING_ROWS.json');targets={x['advice_record_id']:x['target'] for x in rr[37114:]}
    old_inputs={x['record_id']:x for x in rows(DATA/'POLICY_INPUTS.jsonl')}
    episodes={x['index']:x for x in (read(p) for p in FIT.glob('lanes/lane_*/episode_*.json'))}
    states={}
    for folder in sorted(FIT.glob('lanes/lane_*')):
        starts={x['index']:x for x in rows(folder/'INTERFACE.jsonl')}
        steps=collections.defaultdict(list)
        for x in rows(folder/'STEPS_PRIVILEGED.jsonl'):steps[x['index']].append(x)
        for i,start in starts.items():
            frames=[start['rgb_sha256']];actions=[]
            for x in steps[i]:
                t=len(actions);assert x['step']==t+1
                states[i,t]=dict(instruction=episodes[i]['instruction'],rgb_sha256=h['choose'](frames,t),
                    executed_actions=list(actions[-8:]),old_rgb_sha256=frames[-2:])
                frames.append(x['rgb_sha256']);actions.append(x['action'])
    assert len(states)==7225
    view={};provenance={}
    for x in rows(DATA/'SUPERVISION_ONLY.jsonl'):
        key=x['record_id']
        if key not in targets or key in view:continue
        assert x['target']==targets[key] and x['split']=='FIT'
        v=states[x['source_index'],x['decision_step']];oldv=old_inputs[key]
        assert v['instruction']==oldv['instruction'] and v['old_rgb_sha256']==oldv['rgb_sha256']
        assert v['executed_actions']==oldv['executed_actions']
        view[key]=dict(record_id=key,instruction=v['instruction'],rgb_sha256=v['rgb_sha256'],executed_actions=v['executed_actions'])
        provenance[key]=dict(source_index=x['source_index'],episode_id=x['episode_id'],decision_step=x['decision_step'],
            frame_indices=h['indices'](x['decision_step']),first_immutable_occurrence=True)
    assert set(view)==set(targets) and len(view)==4983
    print('RECONSTRUCTED_4983_EXISTING_INPUT_VIEWS',flush=True)
    plan=read(OLD/'PLAN.json');flat=[i for rank in plan for batch in rank for i in batch];wanted=set(flat)
    assert len(flat)==99047 and sum(i<2650347 for i in flat)==89172
    samples={}
    with (OLD/'SAMPLE_INDEX.jsonl').open() as f:
        for i,line in enumerate(f):
            if i in wanted:samples[i]=json.loads(line)
    assert set(samples)==wanted
    ordinary_indices=sorted({x[0] for x in samples.values() if x[0]<37114})
    def load_record(ri):
        row=rr[ri];assert row['source']=='R2R' and row['split']=='FIT'
        root=(ROOT/row['sourceRoot']).resolve(strict=True);assert root.is_relative_to(LINE)
        values={}
        for kind in ('policy','supervision'):
            p=(root/row[kind+'_file']).resolve(strict=True);assert p.is_relative_to(root)
            blob=p.read_bytes();assert hashlib.sha256(blob).hexdigest()==row[kind+'_sha256']
            values[kind]=json.loads(blob)
        return ri,values
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:ordinary=dict(pool.map(load_record,ordinary_indices))
    print('READ_ORIGINAL_R2R_RECORDS',len(ordinary),flush=True)
    fingerprints={};clashes=[];ordinary_changes=0
    A=['move_forward','turn_left','turn_right','STOP']
    normalize=lambda a:dict(MOVE_FORWARD='move_forward',TURN_LEFT='turn_left',TURN_RIGHT='turn_right').get(a,a)
    for i,x in samples.items():
        ri,t,target,weight,est=x
        if ri<37114:
            v=ordinary[ri];p=v['policy'];su=v['supervision'];acts=[normalize(a) for a in su['actions']]
            refs=[Path(x).stem for x in p['rgb_sequence']]
            assert len(refs)==len(acts)==rr[ri]['decisions'] and A[target]==acts[t]
            rgb=h['choose'](refs,t);executed=acts[max(0,t-8):t];instruction=p['instruction']
            assert len(rgb)==min(2,t+1);ordinary_changes+=int(rgb!=refs[max(0,t-1):t+1])
        else:
            row=rr[ri];v=view[row['advice_record_id']];assert target==row['target'] and t==0
            rgb=v['rgb_sha256'];executed=v['executed_actions'];instruction=v['instruction']
        key=h['fingerprint'](instruction,rgb,executed)
        if key in fingerprints and fingerprints[key]!=target:clashes.append(dict(sample_index=i,fingerprint=key,targets=[fingerprints[key],target]))
        fingerprints[key]=target
    save('VIEW_CONFLICT_CHECK.json',dict(status='PASS' if not clashes else 'FAIL',conflicts=clashes,
        checked_planned_unique_indices=len(samples),planned_occurrences=len(flat),unique_input_fingerprints=len(fingerprints)))
    assert not clashes,'NEW_INPUT_VIEW_TARGET_CONFLICT'
    pixels=sorted({x for v in view.values() for x in v['rgb_sha256']})
    def check_pixel(d):
        from PIL import Image
        p=(DATA/'content'/(d+'.png')).resolve(strict=True);assert p.is_relative_to(DATA/'content')
        with Image.open(p) as im:assert im.mode=='RGB' and im.size==(224,224) and hashlib.sha256(im.tobytes()).hexdigest()==d
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check_pixel,pixels))
    save('RECOVERY_VIEW.json',view);save('RECOVERY_PROVENANCE.json',provenance)
    save('VIEW_AUDIT.json',dict(status='PASS',unix=time.time(),stride=8,recovery_source_ids=4983,view_sha256=sha(HERE/'RECOVERY_VIEW.json'),
        recovery_provenance_sha256=sha(HERE/'RECOVERY_PROVENANCE.json'),unique_recovery_pixels_decoded=len(pixels),
        planned_occurrences=99047,ordinary_occurrences=89172,recovery_occurrences=9875,
        planned_unique_sample_indices=len(samples),ordinary_records=len(ordinary),ordinary_inputs_with_different_older_frame=ordinary_changes,
        planned_view_target_conflicts=0,labels_weights_estimates_plan_unchanged=True,encoded_image_counts_unchanged=True,
        new_independent_data_created=0,training_updates=0,simulator_actions=0,
        sample_index_sha256=sha(OLD/'SAMPLE_INDEX.jsonl'),plan_sha256=sha(OLD/'PLAN.json'),
        sources={str(p):sha(p) for p in [OLD/'TRAINING_ROWS.json',DATA/'DATA_SEAL.json',DATA/'SUPERVISION_ONLY.jsonl',
            DATA/'POLICY_INPUTS.jsonl',HERE/'history.py',Path(__file__)]}))
    print('FIXED_PLAN_VIEWS_AND_ACTUAL_PIXEL_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
