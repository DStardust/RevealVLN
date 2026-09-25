"""Review repaired real-action features without selecting checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from review_memory_r1 import review_split


def main(root,phase):
    u.verify_sources(root);torch.set_num_threads(4)
    if phase=='navigation':
        from transfer_review import main as review_navigation
        review_navigation(root/'ordinary',False)
        result=u.read(root/'ordinary/REVIEW.json')
        u.write(root/'RESULT.json',dict(status='COMPLETE',action_interface='REAL_ACTION_POSITION_V2',
            navigation=result,training=u.read(root/'TRAINING_REVIEW.json'),
            interpretation='FIT fitting, ordinary preservation and recovery efficacy are distinct; no history recovery closed-loop result'))
        (root/'REPORT_ZH.md').write_text('REAL_ACTION_POSITION_V2\n\n'+(root/'ordinary/REPORT_ZH.md').read_text())
        return
    data=root/'data';training=root/'training';states={};receipts={};binding=None
    admission=u.read(data/'ADMISSION.json')
    schedule=u.read(training/'SCHEDULE_42.json');assert len(schedule)==1200
    for arm in ('BC','B2','OURS'):
        path=training/arm/'FINAL.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        receipt=u.read(training/arm/'RESULT.json');assert u.sha(path)==receipt['final_sha256']
        assert saved['step']==1200 and saved['arm']==arm and receipt['parameters_changed']
        assert saved['binding']['base_updates']==0 and saved['binding']['data_sha256']==admission['feature_shas']['FIT']
        if binding is None:binding=saved['binding']
        else:assert binding==saved['binding'],'TRAINING_ARMS_NOT_MATCHED'
        rows=[json.loads(s) for s in (training/arm/'STEPS.jsonl').read_text().splitlines()]
        # Resume uses saved optimizer/RNG; latest attempt supplies each restored
        # step. Earlier uncheckpointed executions remain visible in the journal.
        final={r['step']:r for r in rows};assert set(final)==set(range(1,1201))
        assert [final[i]['group'] for i in range(1,1201)]==schedule
        receipts[arm]=dict(steps=1200,executed_optimizer_steps_including_replayed=len(rows),weight_sha256=u.sha(path),
            final_loss=final[1200]['loss'],parameters_changed=True)
        states[arm]=saved['model']
    results={}
    for split in ('FIT','CHECK'):
        assert u.sha(data/(split+'.pt'))==admission['feature_shas'][split]
        pack=torch.load(data/(split+'.pt'),map_location='cpu',weights_only=True)
        assert pack['action_boundary']=='AFTER_NATIVE_ASSISTANT_HEADER_V2'
        results[split]=review_split(pack,states)
    u.write(root/'TRAINING_REVIEW.json',dict(status='TRAINING_COMPLETE',training=receipts,results=results,
        action_boundary='AFTER_NATIVE_ASSISTANT_HEADER_V2',base_updates=0,
        inference_not_from_header_token=True,scope='FIT action fitting and admitted CHECK only; not closed-loop efficacy'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['training','navigation'],required=True);a=p.parse_args()
    main(a.root,a.phase)

