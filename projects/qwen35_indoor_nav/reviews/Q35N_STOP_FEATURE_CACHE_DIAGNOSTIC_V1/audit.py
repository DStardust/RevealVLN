import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main():
    import torch
    torch.set_num_threads(4);c.verify();parent=c.LINE/'sft_acceptance/ordinary_stop_row_v12'
    labels={x['record_id']:x for x in c.rows(parent/'SUPERVISION_ONLY.jsonl')};data=torch.load(HERE/'FEATURES.pt',map_location='cpu',weights_only=True)
    expected=torch.tensor([labels[k]['occurrences'][0]['original_logits'] for k in data['record_ids']]);old=torch.load(parent/'FEATURES.pt',map_location='cpu',weights_only=True)
    keys={x:i for i,x in enumerate(old['record_ids'])};fresh=old['logits'][[keys[k] for k in data['record_ids']]]
    diff=float((data['logits']-expected).abs().max());same=bool(torch.equal(data['logits'].argmax(1),expected.argmax(1)))
    cache=c.read(HERE/'CACHE_COPY.json')['files'];assert all(c.sha(p)==h for p,h in cache.items())
    result=dict(status='PASS' if diff<=1e-5 and same else 'FAIL',unix=time.time(),inputs=32,max_abs_original=diff,original_argmax_same=same,fresh_cache_max_abs_original=float((fresh-expected).abs().max()),copied_cache_unchanged=True,new_autotune_keys=[str(p) for p in (HERE/'triton_cache').rglob('*.autotune.json') if str(p) not in cache],navigation_gain=False,parameter_updates=0)
    c.write(HERE/'RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
