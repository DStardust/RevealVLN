"""Fixed final-step FIT/DEV gate; no selection over steps, weights or new houses."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def assess(diagnoses,maximum_ordinary_accuracy_drop=.005):
    def pool(split,key,field):return sum(r[key][field] for rows in diagnoses for r in rows if r['split']==split)
    def improvement(rows):return sum(r['old']['correct_state_missed_stop']-r['repair']['correct_state_missed_stop'] for r in rows if r['split']=='FIT')
    old=pool('ORDINARY_CHECK','old','correct');new=pool('ORDINARY_CHECK','repair','correct');n=pool('ORDINARY_CHECK','old','n')
    if not n:raise ValueError('EMPTY_ORDINARY_CHECK')
    checks=dict(fit_correct_state_missed_stop_reduced=pool('FIT','repair','correct_state_missed_stop')<pool('FIT','old','correct_state_missed_stop'),
        fit_improvement_at_least_two_seeds=sum(improvement(rows)>0 for rows in diagnoses)>=2,
        fit_no_seed_regression=all(improvement(rows)>=0 for rows in diagnoses),
        dev_missed_stop_reduced=pool('DEV','repair','missed_stop')<pool('DEV','old','missed_stop'),
        dev_false_stop_not_increased=pool('DEV','repair','false_stop')<=pool('DEV','old','false_stop'),
        ordinary_accuracy_drop_within_limit=(old-new)/n<=maximum_ordinary_accuracy_drop,
        ordinary_false_stop_not_increased=pool('ORDINARY_CHECK','repair','false_stop')<=pool('ORDINARY_CHECK','old','false_stop'))
    return dict(admit_closed_loop=all(checks.values()),checks=checks,ordinary_accuracy_delta=(new-old)/n,
                dev_missed_stop=dict(old=pool('DEV','old','missed_stop'),repair=pool('DEV','repair','missed_stop')),
                dev_false_stop=dict(old=pool('DEV','old','false_stop'),repair=pool('DEV','repair','false_stop')))

def main(run):
    cfg=config(run);values=[];identities=[]
    for seed in cfg['seeds']:
        folder=run/'train'/f'STOPFIX_{seed}';record=read(folder/'RESULT.json')
        if record['updates']!=cfg['repair_updates'] or not record['frozen_model_unchanged'] or not record['readout_changed'] or record['base_updates'] or record['original_model_updates']:
            raise ValueError('INVALID_READOUT_TRAINING')
        if sha(folder/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('FINAL_CHANGED')
        values.append(read(folder/'LOCAL_DIAGNOSIS.json')['rows']);identities.append(record)
    result=dict(assess(values,cfg['local_gate']['max_ordinary_accuracy_drop']),seeds=cfg['seeds'],diagnoses=values,identities=identities,
        scope='Exposed FIT/DEV local diagnostic only; not independent navigation effectiveness')
    immutable(run/'LOCAL_GATE.json',result)
    if not result['admit_closed_loop']:
        immutable(run/'RESULT.json',dict(status='LOCAL_REPAIR_NOT_ADMITTED',complete=0,planned=cfg['planned_rollouts'],
            new_updates=cfg['repair_updates']*len(cfg['seeds']),base_updates=0,original_model_updates=0,method_adopted=False,
            closed_loop_effect='UNKNOWN_NOT_RUN',local_gate=result))
        (run/'REPORT_ZH.md').write_text('LOCAL_REPAIR_NOT_ADMITTED\n\n三个种子均完成固定1200步STOP读出训练；原基座和记忆更新0。闭环0/768，未将未运行计作失败。\n\n'
            +'局部检查：'+str(result['checks'])+'\n\n训练代码、真实梯度、参数更新和所有诊断保留。未通过预注册局部检查，不自动调参重跑，不采用为方法收益。原数据和历史结果保持不变。\n')

if __name__=='__main__':main(Path(sys.argv[1]))
