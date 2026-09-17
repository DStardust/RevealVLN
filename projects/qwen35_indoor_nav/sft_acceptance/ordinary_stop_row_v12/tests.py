import ast,hashlib,importlib.util,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
import torch
from PIL import Image
def main():
    assert not (HERE/'CPU_TEST_RESULT.json').exists()
    f=c.load('fit_module',HERE/'fit.py');labels=c.rows(HERE/'SUPERVISION_ONLY.jsonl');inputs=c.rows(HERE/'POLICY_INPUTS.jsonl');split=c.read(HERE/'SPLIT.json')
    assert [x['record_id'] for x in labels]==[x['record_id'] for x in inputs]
    assert len(inputs)==5487 and sum(x['target'] for x in labels)==818
    assert set(split['fit_houses']).isdisjoint(split['heldout_houses']) and len(split['fit_houses'])==12 and len(split['heldout_houses'])==4
    w=f.weights(labels,split['fit_houses']);v=f.weights(labels,split['heldout_houses']);assert not ((w>0)&(v>0)).any()
    assert abs(w.sum()-1)<1e-12 and abs(v.sum()-1)<1e-12
    hashes=set()
    for x in inputs:
        assert set(x)=={'record_id','instruction','rgb_sha256','executed_actions'}
        assert 1<=len(x['rgb_sha256'])<=2 and len(x['executed_actions'])<=8
        hashes.update(x['rgb_sha256'])
    assert len(hashes)==5434
    for digest in hashes:
        with Image.open(c.DATA/'content'/(digest+'.png')) as im:
            rgb=im.convert('RGB');assert rgb.size==(224,224) and hashlib.sha256(rgb.tobytes()).hexdigest()==digest
    torch.manual_seed(1209);h=torch.randn(64,8,dtype=torch.float64);z=torch.randn(64,4,dtype=torch.float64);y=(h[:,0]>0).double();ww=torch.full((64,),1/64,dtype=torch.float64)
    theta,scale,info=f.optimize(h,z,y,ww);old=z[:,3]-z[:,:3].max(1).values;new=old+h@theta[:-1]/scale+theta[-1]
    assert f.metrics(new,y,ww)['bce']<f.metrics(old,y,ww)['bce']
    before=torch.randn(4,8);bias=torch.randn(4);after=before.clone();newbias=bias.clone();after[3]+=theta[:-1].float()/scale.float();newbias[3]+=theta[-1].float()
    assert torch.equal(before[:3],after[:3]) and torch.equal(bias[:3],newbias[:3])
    a=h.float()@before.T+bias;b=h.float()@after.T+newbias;assert torch.equal(a[:,:3],b[:,:3])
    assert torch.allclose(b[:,3].double(),a[:,3].double()+h@theta[:-1]/scale+theta[-1],atol=1e-5,rtol=1e-5)
    assert f.metrics(torch.zeros(1),torch.ones(1),torch.ones(1))['recall']==0
    original=torch.load(c.BEST,map_location='cpu',weights_only=True)
    assert original['cursor']['updates']==4000 and original['trainable']['action_head.weight'].dtype==torch.float32
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    c.write(HERE/'CPU_TEST_RESULT.json',dict(status='PASS',unix=time.time(),actual_inputs=5487,pngs_pixel_hashed=5434,house_split_disjoint=True,actual_weight_normalization=True,linear_fit_synthetic=True,folded_head_equivalence=True,motion_rows_bit_exact=True,tie_not_stop=True,best_checkpoint_schema=True,code_ast=True,extra_gpu_forwards=0,tests_code_sha256=c.sha(Path(__file__))))
    print(json.dumps(c.read(HERE/'CPU_TEST_RESULT.json')))
if __name__=='__main__':main()
