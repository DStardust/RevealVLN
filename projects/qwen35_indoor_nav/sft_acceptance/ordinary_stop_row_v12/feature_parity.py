import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main():
    import torch
    torch.set_num_threads(4);c.verify()
    cache=torch.load(HERE/'FEATURES.pt',map_location='cpu',weights_only=True);labels=c.rows(HERE/'SUPERVISION_ONLY.jsonl')
    assert cache['record_ids']==[x['record_id'] for x in labels]
    expected=torch.tensor([x['occurrences'][0]['original_logits'] for x in labels])
    actual=cache['logits'];diff=float((actual-expected).abs().max());same=bool(torch.equal(actual.argmax(1),expected.argmax(1)))
    status='PASS' if diff<=1e-5 and same else 'FAIL'
    c.write(HERE/'FEATURE_PARITY.json',dict(status=status,unix=time.time(),max_abs=diff,all_original_argmax_same=same,inputs=len(labels),feature_sha256=c.sha(HERE/'FEATURES.pt'),no_feature_inputs_contain_supervision=True,extra_forwards=0))
    print(json.dumps(c.read(HERE/'FEATURE_PARITY.json')));assert status=='PASS'
if __name__=='__main__':main()
