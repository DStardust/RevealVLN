"""Read-only checkpoint diagnosis: deployed STOP guard and learned exact state."""
from pathlib import Path
import sys
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import objective as o
c=o.c


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    train=HERE/'train_run_001';assert len(c.read(train/'RESULT.json')['runs'])==18
    data=c.read(HERE/'DATA.json');cache=torch.load(HERE/'features_run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    batches=[o.batch(f) for f in data['families']];rows=[]
    with torch.inference_mode():
        for run in c.read(train/'RESULT.json')['runs']:
            recorded=c.read(train/run['result']);net=o.initialize(recorded['seed'])
            net.load_state_dict(torch.load(train/(recorded['tag']+'_MEMORY.pt'),map_location='cpu',weights_only=True));net.eval()
            assert c.model_identity(net)['sha256']==recorded['final_state_sha256']
            families=[]
            for b in batches:
                states,_=net.encode(cache['features'][b['indices']]);final=states[torch.arange(len(b['lengths'])),b['lengths']-1]
                guarded={mode:0 for mode in ('correct','wrong','sham')};native_stop_forks=0
                for case in b['forks']:
                    base=cache['logits'][case['common_feature']];native=int(base.argmax())
                    native_stop_forks+=int(native==3)
                    target=[case['left_action'],case['right_action']]
                    for mode in guarded:
                        logits=o.fork_logits(net,cache,b,final,case,mode)
                        chosen=[3,3] if native==3 else logits.argmax(-1).tolist()
                        guarded[mode]+=int(chosen==target)
                exact=[]
                if recorded['arm']=='B2':
                    predicted=net.state_head(states.flatten(2))>0;target=b['state_targets'].bool()
                    for bit,name in enumerate(('anchor_before','anchor_including_now','terminal_now','ready_to_stop')):
                        positives=b['mask'] & target[:,:,bit];negatives=b['mask'] & ~target[:,:,bit]
                        tp=int((predicted[:,:,bit]&positives).sum());tn=int((~predicted[:,:,bit]&negatives).sum())
                        npos=int(positives.sum());nneg=int(negatives.sum())
                        exact.append(dict(name=name,positive=npos,negative=nneg,tp=tp,tn=tn,
                            balanced_accuracy=(tp/npos+tn/nneg)/2 if npos and nneg else None))
                families.append(dict(family_id=b['family']['family_id'],split=b['family']['split'],
                    guarded_fork_both_correct=guarded,fork_pairs=len(b['forks']),native_stop_forks=native_stop_forks,
                    learned_exact_state=exact))
            assert c.model_identity(net)['sha256']==recorded['final_state_sha256']
            rows.append(dict(tag=recorded['tag'],families=families));print(recorded['tag'],flush=True)
    c.write(HERE/'HEAD_DIAGNOSIS.json',dict(rows=rows,optimizer_updates=0,GPU_used=False,
        diagnostic_scope='Frozen cached-feature checkpoint readout, not online task success; exact state head is trained only for B2.'),True)


if __name__=='__main__':main()
