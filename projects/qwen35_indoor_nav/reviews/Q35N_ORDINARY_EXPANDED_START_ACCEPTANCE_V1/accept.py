"""Main-agent read-only live/checkpoint acceptance, outputs only this review."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_expanded_v1'
EXP=LINE/'data_pipeline/ordinary_expansion_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def identity(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=raw[0],start=int(raw[19]),cwd=str((p/'cwd').resolve()),
                argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ACCEPTANCE_ONLY'
    import torch
    protocol=read(TRAIN/'PROTOCOL_FILESTORE.json')
    for name,digest in protocol['code_sha256'].items():assert sha(TRAIN/name)==digest,('CODE_CHANGED',name)
    snap=EXP/'training_snapshot_20260912_v1/run_001'
    for name,digest in read(snap/'SEAL.json').items():assert sha(snap/name)==digest
    assert sha(EXP/'runtime_v2/INPUT_LOCK.json')=='c693ff433c3a02557c5d125400ce77d877b396a4d26544ffd7420fba1189509f'
    assert sha(EXP/'runtime_gpu3_recovery_v1/INPUT_LOCK.json')=='fcb9eb183ccf48b66e6a532b55c23550b96c4c76312cdc42a309bae73aedf6f4'
    status=read(TRAIN/'formal/STATUS.json');progress=read(Path(status['run_dir'])/'PROGRESS.json')
    assert status['status']=='TRAINING' and time.time()-status['unix']<20 and time.time()-progress['unix']<60
    assert progress['cursor']['updates']>=200
    checkpoint=Path(status['run_dir'])/'checkpoint_000000200.pt';receipt=read(str(checkpoint)+'.json')
    assert sha(checkpoint)==receipt['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    initial=torch.load(TRAIN/'initial_from_51301.pt',map_location='cpu',weights_only=True)
    assert state['cursor']==receipt['cursor'] and state['cursor']['updates']==200
    assert state['binding']['protocol_sha256']==sha(TRAIN/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==protocol['sample_index_sha256']
    assert state['global_decisions']==state['charged_compute_decisions']==receipt['global_decisions']>0
    changed=[]
    for key,value in state['trainable'].items():
        assert torch.isfinite(value).all()
        if not torch.equal(value,initial['trainable'][key]):changed.append(key)
    assert changed and state['optimizer']['state']
    for slot in state['optimizer']['state'].values():
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    metrics=progress['metrics']
    assert math.isfinite(metrics['mean_ce']) and math.isfinite(metrics['grad_norm']) and metrics['grad_norm']>0
    raw=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15)
    gpu_rows=[]
    for i,g in enumerate(ET.fromstring(raw).findall('gpu')):
        contexts=[dict(pid=int(x.findtext('pid')),type=x.findtext('type'),used=x.findtext('used_memory')) for x in g.findall('processes/process_info')]
        gpu_rows.append(dict(index=i,uuid=g.findtext('uuid'),memory=g.findtext('fb_memory_usage/used'),contexts=contexts))
    ranks=[]
    for gpu in gpu_rows[3:6]:
        assert gpu['uuid']==protocol['gpus'][gpu['index']-3]
        assert len(gpu['contexts'])==1
        own=identity(gpu['contexts'][0]['pid']);assert str(TRAIN/'train_filestore.py') in own['argv'];ranks.append(own)
    protected=read(LINE/'sft_acceptance/monitor_charts_expanded_v1/DEPLOY_R2_BEFORE.json')['protected']
    for pid in (3996270,112240):
        live=identity(pid);expected=protected[str(pid)]
        assert all(live[k]==expected[k] for k in ('pid','start','cwd','argv'))
    with urllib.request.urlopen('http://127.0.0.1:18766/api/status',timeout=15) as f:monitor=json.load(f)
    assert monitor['monitor_version']=='ordinary_expanded_v1' and monitor['display_state']=='TRAINING'
    assert monitor['snapshot_counts']['instruction_conditioned_decisions']==2650347
    assert all(x['segment']!='legacy_v3' for x in monitor['points'])
    history=[json.loads(x) for x in (Path(status['run_dir'])/'PROGRESS.jsonl').read_text().splitlines()]
    warmed=[r['throughput'] for r in history if r['cursor']['updates']>=80]
    result=dict(status='LIVE_TRAINING_AND_CHECKPOINT_ACCEPTED',unix=time.time(),snapshot_counts=read(snap/'RESULT.json')['counts'],
         training_index_sha256=protocol['snapshot']['training_index_sha256'],sample_index_sha256=protocol['sample_index_sha256'],
         source_checkpoint_updates=51301,new_stage_updates=progress['cursor']['updates'],new_stage_decisions=progress['global_plan_decisions'],
         accepted_checkpoint=str(checkpoint),accepted_checkpoint_sha256=receipt['sha256'],checkpoint_parameters_changed=len(changed),
         all_checkpoint_tensors_finite=True,code_and_old_input_locks_unchanged=True,ranks=ranks,gpu_snapshot=gpu_rows,
         recent_measured_throughput_range=[min(warmed),max(warmed)],monitor_port=18766,monitor_data_counts_correct=True,
         gpu6_gpu7_holder_identities_unchanged=True,current_gpu3_gpu4_gpu5_restoration_pending=True,
         navigation_gain_verified=False,architecture_innovation_claimed=False,
         development_comparison_status=read(LINE/'closed_loop_bench/ordinary_expanded_dev_pair_v1/STATUS.json')['status'])
    with (HERE/'RESULT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','new_stage_updates','new_stage_decisions','checkpoint_parameters_changed','recent_measured_throughput_range')}))


if __name__=='__main__':main()
