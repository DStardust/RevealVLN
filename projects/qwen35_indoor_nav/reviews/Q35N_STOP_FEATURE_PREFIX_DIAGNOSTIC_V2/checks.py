import hashlib,json,time
from pathlib import Path
def check(model,policy,collate,forward,c,stage):
    import torch
    from PIL import Image
    parent=c.LINE/'sft_acceptance/ordinary_stop_row_v12';data=c.LINE/'data_pipeline/ordinary_route_teacher_v11/run_001/content'
    allrows=[json.loads(x) for x in (parent/'POLICY_INPUTS.jsonl').read_text().splitlines()];indices=list(range(16))+[300*i for i in range(1,17)];rows=[allrows[i] for i in indices]
    class Store:
        def __init__(self,mode):self.mode=mode
        def get(self,i,t):
            row=rows[i];images=[];raw=[]
            for sha in row['rgb_sha256']:
                with Image.open(data/(sha+'.png')) as im:im=im.convert('RGB');images.append(im);raw.append(im.tobytes())
            if self.mode=='png':return dict(instruction=row['instruction'],executed=row['executed_actions'],images=images)
            window=c.Window();window.instruction=row['instruction'];window.executed=row['executed_actions'];window.images=raw;return window.item()
    samples=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(32)]
    a=model.DecisionDataset(samples,Store('png'),policy.processor);b=model.DecisionDataset(samples,Store('window'),policy.processor)
    encoded=[a[i] for i in range(32)];other=[b[i] for i in range(32)]
    for x,y in zip(encoded,other):
        xx=collate([x]);yy=collate([y]);assert xx.keys()==yy.keys() and all(torch.equal(xx[k],yy[k]) for k in xx)
    with torch.inference_mode():
        normal=torch.cat([forward([x]).cpu() for x in encoded])
        captured=[]
        def hook(module,args):captured.append(args[0].detach().cpu().clone())
        handle=policy.action_head.register_forward_pre_hook(hook)
        try:hooked=torch.cat([forward([x]).cpu() for x in encoded])
        finally:handle.remove()
    assert len(captured)==32
    labels={x['record_id']:x for x in (json.loads(t) for t in (parent/'SUPERVISION_ONLY.jsonl').read_text().splitlines())}
    expected=torch.tensor([labels[x['record_id']]['occurrences'][0]['original_logits'] for x in rows])
    old=torch.load(parent/'FEATURES.pt',map_location='cpu',weights_only=True)['logits'][indices]
    result=dict(stage=stage,indices=indices,normal=normal.tolist(),hooked=hooked.tolist(),old_logits=expected.tolist(),max_abs_old=float((normal-expected).abs().max()),max_abs_v12=float((normal-old).abs().max()),hook_max_abs=float((normal-hooked).abs().max()),cpu_encoding_exact=True,original_argmax_same=bool(torch.equal(normal.argmax(1),expected.argmax(1))))
    c.write(c.HERE/'run_001'/('DIAGNOSTIC_'+stage.upper()+'.json'),result,True)
    return result
def finish(early,late,c,fingerprint):
    import torch
    diff=float((torch.tensor(early['normal'])-torch.tensor(late['normal'])).abs().max())
    result=dict(status='PASS' if late['max_abs_old']<=1e-5 and late['original_argmax_same'] else 'FAIL',unix=time.time(),before={k:v for k,v in early.items() if k not in ('normal','hooked','old_logits')},after={k:v for k,v in late.items() if k not in ('normal','hooked','old_logits')},prewarmup_max_abs=diff,trainable_fingerprint=fingerprint,model_unchanged=True,forward_decisions=160,simulator_actions=0,parameter_updates=0,navigation_gain=False)
    c.write(c.HERE/'RESULT.json',result,True);print(json.dumps(result),flush=True)
