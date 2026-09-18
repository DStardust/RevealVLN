"""CPU read-only confusion matrices and a constant-residual mechanism probe.

The constant is the mean learned residual on FIT inputs, without fitting labels.
It is never sent to a simulator or selected as a deployment policy.
"""
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


def confusion(logits,targets,native):
    prediction=torch.where(native.argmax(-1)==3,3,logits.argmax(-1))
    matrix=torch.bincount(targets*4+prediction,minlength=16).reshape(4,4)
    return dict(matrix_true_rows_predicted_columns=matrix.tolist(),
        accuracy=float(matrix.diag().sum()/matrix.sum()),
        class_recall=(matrix.diag()/matrix.sum(1)).tolist(),
        predictions=matrix.sum(0).tolist(),decisions=int(matrix.sum()))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    run=HERE/'run_001';config=c.read(HERE/'PROTOCOL.json');data=c.read(HERE/'DATA.json')
    identity=c.read(run/'FEATURE_RESULT.json')
    assert c.sha(run/'FEATURES.pt')==identity['file_sha256']
    cache={k:v.float() for k,v in torch.load(run/'FEATURES.pt',map_location='cpu',weights_only=True).items()}
    special=c.read(HERE.parent/'multifamily_v7/DATA.json');result=c.read(run/'RESULT.json')
    records=data['records'];rows={}
    for seed in config['seeds']:
        for arm in config['arms']:
            key=f'{arm}_{seed}'
            net=train.models.MemoryPolicy(2048,len(special['query_vocabulary']),8,64,.99).eval()
            net.load_state_dict(torch.load(run/f'{key}_MEMORY.pt',map_location='cpu',weights_only=True))
            expected=result['runs'][key]['final_state_sha256']
            assert c.model_identity(net)['sha256']==expected
            fit_means=[];check_logits=[];check_native=[];check_targets=[]
            for start in range(0,len(records),8):
                selected=records[start:start+8];batch=train.ordinary_batch(selected,'cpu')
                _,logits=train.ordinary_loss(net,cache,batch)
                for i,row in enumerate(selected):
                    n=len(row['features']);native=cache['logits'][batch['indices'][i,:n]]
                    if row['partition']=='fit':fit_means.append((logits[i,:n]-native).mean(0))
                    else:
                        check_logits.append(logits[i,:n]);check_native.append(native);check_targets.append(batch['targets'][i,:n])
                c.write(HERE/'ACTION_DIAGNOSIS_PROGRESS.json',dict(unix=time.time(),key=key,completed_routes=min(start+8,len(records)),total_routes=len(records)))
            residual=torch.stack(fit_means).mean(0)
            logits=torch.cat(check_logits);native=torch.cat(check_native);target=torch.cat(check_targets)
            assert len(fit_means)==508 and len(target)==3769
            assert c.model_identity(net)['sha256']==expected
            rows[key]=dict(native=confusion(native,target,native),memory=confusion(logits,target,native),
                constant_fit_residual=confusion(native+residual,target,native),mean_fit_residual=residual.tolist(),
                parameters_unchanged=True,fit_labels_used_to_compute_constant=False)
            print(key,{k:rows[key][k]['accuracy'] for k in ('native','memory','constant_fit_residual')},flush=True)
    c.write(HERE/'ACTION_DIAGNOSIS.json',dict(status='READ_ONLY_TEACHER_ACTION_MECHANISM_PROBE',runs=rows,
        source_sha256=c.sha(Path(__file__)),gpu_used=False,optimizer_updates=0,new_navigation_episodes=0,
        actions=['move_forward','turn_left','turn_right','STOP'],
        interpretation='A mean-residual probe measures how much teacher-action improvement needs input-dependent memory. It neither proves nor disproves semantic history use, and is not a navigation policy trial.'),True)


if __name__=='__main__':main()
