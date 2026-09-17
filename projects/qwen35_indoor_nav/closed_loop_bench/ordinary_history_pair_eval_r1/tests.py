"""Actual window/aggregate integration and malicious-input rejection; CPU only."""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASES={'control_recent2':'ordinary_history2_dev_r1','treatment_prefix8':'ordinary_history8_dev_r1'}
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    audit=runpy.run_path(str(HERE/'history_audit.py'))
    shared=runpy.run_path(str(HERE/'reuse.py'))
    runtime=runpy.run_path(str(LINE/'sft_acceptance/ordinary_history8_paired_train_r1/runtime.py'))
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    data=runtime['ordinary']();rows,_=data.load_rows()
    store=data.SampleStore(rows)
    groups={}
    for i,row in enumerate(rows):
        if row['source'] not in groups and 16<=row['decisions']<=200:groups[row['source']]=i
    assert len(groups)==3
    model=runtime['load']('history_eval_cpu_model',LINE/'sft_acceptance/ordinary_history8_paired_train_r1/model.py')
    processor=AutoProcessor.from_pretrained(model.MODEL,local_files_only=True,trust_remote_code=False)
    ids=json.loads((LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens':list(ids)})
    image_id=json.loads((model.MODEL/'config.json').read_text())['image_token_id']
    collate=model.make_collate(processor.tokenizer.pad_token_id,{a:ids[model.EXEC_TOKENS[a]] for a in model.ACTIONS[:-1]},ids['<NAV_ACTION_QUERY>'],image_id)
    results={};rejected=0
    for arm,dirname in CASES.items():
        case=HERE.parent/dirname
        m=runpy.run_path(str(case/'common.py'),run_name='CPU_COMMON')
        assert m['TRAIN']==LINE/'sft_acceptance/ordinary_history8_paired_train_r1'
        for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
            text=shared['source'](name,arm);ast.parse(text)
            if name in ('executor.py','path_metrics.py','official_distance.py'):
                assert text==shared['parent'].source(name)
        evaltext=shared['source']('evaluate.py',arm)
        assert 'batch_size=1  # Matched comparison' in evaltext and "state['binding']['arm']==p['history_arm']" in evaltext
        assert 'stop_logit_bias' not in evaltext
        launch=runpy.run_path(str(case/'launch.py'),run_name='CPU_LAUNCH_IMPORT')
        assert launch['OUT']==case/'run_001'
        traced=0;tensors=0
        for idx in groups.values():
            record=store.record(idx);window=m['Window']();observed=[]
            offline=runtime['stores'](rows)[arm]
            for t in range(len(record)):
                rgb=data.decoder['load_rgb'](data.decoder['_relative'](record.rgb_root,record._refs[t])).tobytes()
                observed.append(hashlib.sha256(rgb).hexdigest())
                payload=dict(done=False,rgb=base64.b64encode(rgb).decode())
                if t==0:payload['instruction']=record.instruction
                assert window.receive(payload,None if t==0 else record.actions[t-1])
                pol=dict(images=len(window.images),executed_history=len(window.executed),**window.input_audit())
                audit['validate'](arm,pol,dict(step=t+1),observed)
                traced+=1
                if t in (0,1,7,8,len(record)//2,len(record)-1):
                    for key,bad in [('input_frame_indices',[t+1]*len(window.images)),
                                    ('input_rgb_sha256',['f'*64]*len(window.images)),
                                    ('padding_mask',[True]*len(window.images)),('history_rule','wrong'),
                                    ('images',999),('executed_history',99)]:
                        broken=copy.deepcopy(pol);broken[key]=bad
                        try:audit['validate'](arm,broken,dict(step=t+1),observed)
                        except AssertionError:rejected+=1
                        else:raise AssertionError('CORRUPTION_ACCEPTED:'+key)
                    online=window.item()
                    class Online:
                        def get(self,*args):return online
                    sample=[dict(record_idx=idx,t=t,target=0,weight=1.)]
                    a=model.DecisionDataset(sample,offline,processor)
                    b=model.DecisionDataset(sample,Online(),processor)
                    aa=collate([a[0]]);bb=collate([b[0]])
                    assert set(aa)==set(bb) and all(torch.equal(aa[k],bb[k]) for k in aa)
                    tensors+=1
            assert not window.receive(dict(done=True),'STOP')
            assert window.receive(dict(done=False,rgb=base64.b64encode(rgb).decode(),instruction='new episode'))
            pol=dict(images=len(window.images),executed_history=len(window.executed),**window.input_audit())
            audit['validate'](arm,pol,dict(step=1),[hashlib.sha256(rgb).hexdigest()])
        results[arm]=dict(real_record_indices=list(groups.values()),actual_trace_inputs=traced,
                          actual_offline_online_tensor_comparisons=tensors,reset_and_terminal_passed=True)
    for t in range(500):
        indices=audit['expected_indices']('treatment_prefix8',t)
        assert indices==runpy.run_path(str(LINE/'sft_acceptance/ordinary_prefix_history8_v1/history.py'))['indices'](t)
        assert all(i is None or 0<=i<=t for i in indices)
    assert not torch.cuda.is_initialized()
    for path in list(HERE.glob('*.py'))+sum((list((HERE.parent/x).glob('*.py')) for x in CASES.values()),[]):ast.parse(path.read_text())
    result=dict(passed=True,status='PASS_ACTUAL_WINDOW_AND_AUDIT_CPU',unix=time.time(),arms=results,
                corruption_rejections=rejected,all_500_online_indices_checked=True,physics_and_metric_source_unchanged=True,
                training_updates=0,gpu_launches=0,simulator_actions=0,
                sources={str(p):runtime['sha'](p) for p in list(HERE.glob('*.py'))+sum((list((HERE.parent/x).glob('*.py')) for x in CASES.values()),[])})
    runtime['save'](HERE/'CPU_TEST_RESULT.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
if __name__=='__main__':main()

