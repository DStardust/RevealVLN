"""Localize with held-out readouts and unchanged policies; no navigation claims."""
import sys,statistics
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *


def main():
    cfg=read(HERE/'PROTOCOL.json');diag=read(OUT/'DATA.json');data=read(RUN/'DATA.json');visits=diag['visits'];families={f['family_id']:f for f in data['families']}
    event=[];memory=[];state=[];actions=[];retention=[];gradient=[];resources=[]
    for seed in cfg['seeds']:
        folder=OUT/f'seed_{seed}';resources.append(read(folder/'RESULT.json'))
        for kind,value in read(folder/'EVENT_PROBE.json').items():
            event.extend(dict(seed=seed,kind=kind,**r) for r in value['tables'])
        for arm,value in read(folder/'MEMORY_PROBE.json').items():
            memory.extend(dict(seed=seed,arm=arm,**r) for r in value['tables'])
        gradient.extend(dict(seed=seed,**r) for r in read(folder/'GRADIENT_CONFLICT.json')['rows'])
        for arm in ('B1','B2','Terminal'):
            records=read(folder/(arm+'_POLICY_ROWS.json'))
            for split in ('FIT','DEV'):
                for task in ('task_A','task_T'):
                    ids=[i for i,v in enumerate(visits) if v['split']==split and v['task']==task]
                    for phase in ('all','first_witness','never','current','1-8','9-16','17+'):
                        q=[i for i in ids if phase=='all' or (visits[i]['first_witness'] if phase=='first_witness' else age_bucket(visits[i]['age'])==phase)]
                        for bit in (range(4) if arm=='B2' else (2,) if arm=='Terminal' else ()):
                            state.append(dict(seed=seed,arm=arm,split=split,task=task,phase=phase,bit=bit,
                                **metrics([visits[i]['state'][bit] for i in q],[records[i]['probability'][bit] for i in q])))
                    teacher=[i for i in ids if visits[i]['action_mask']]
                    for correctness in ('all','state_correct','state_wrong') if arm=='B2' else ('all',):
                        q=[i for i in teacher if correctness=='all' or records[i]['state_correct']==(correctness=='state_correct')]
                        actions.append(dict(seed=seed,arm=arm,split=split,task=task,state=correctness,n=len(q),
                            any_registered_teacher_agreement=sum(records[i]['teacher_agreement'] for i in q),
                            unsafe_stop=sum(records[i]['prediction']==3 and not visits[i]['state'][3] for i in q),
                            continue_when_ready=sum(records[i]['prediction']!=3 and visits[i]['state'][3] for i in q),
                            native_teacher_conflict=sum(records[i]['native'] not in visits[i]['teacher_targets'] for i in q),
                            native_teacher_conflict_under_KL=sum(records[i]['native'] not in visits[i]['teacher_targets'] and visits[i]['preservation'] for i in q)))
            if arm=='B2':
                done=set()
                for seq in diag['sequences']:
                    f=families[seq['family']];r=f['sequences'][seq['sequence']]
                    if r['task']!='task_A':continue
                    ids=seq['visit_ids'];cut=ids[r['cutoff']]
                    if cut in done:continue
                    done.add(cut);first=next((i for i in ids[:r['cutoff']+1] if visits[i]['first_witness']),None)
                    if first is None:continue
                    retention.append(dict(seed=seed,split=f['split'],house=f['house'],parent=f['parent_family_id'],cutoff_row=cut,
                        first_witness_row=first,age_at_cutoff=visits[cut]['age'],
                        detected_first=records[first]['probability'][1]>=.5,retained_at_cutoff=records[cut]['probability'][1]>=.5))
    result=dict(status='COMPLETE_LOCALIZATION',event_probes=event,memory_probes=memory,state_diagnosis=state,teacher_action_diagnosis=actions,
        conditional_retention=retention,gradient_conflicts=gradient,resources=resources,
        limitations=['One exposed DEV house; no independent test or navigation improvement.',
        'Probes are diagnostic-only and trained on FIT; failure is not proof of absent information.',
        'Teacher-path state correctness is not autonomous accuracy; teacher action mismatch may have other legal alternatives.',
        'Gradient cosine is a local conflict signal, not evidence that removing a loss improves navigation.',
        'Full-state auxiliary and actor share recurrent parameters, but the actor does not directly consume state_head outputs.'],
        new_navigation_episodes=0,policy_updates=0,base_updates=0)
    immutable(OUT/'RESULT.json',result)
    lines=['# B2 冻结特征定位','',result['status'],'','零新导航、零策略更新、零底模前向。探针只在 FIT 训练，固定 final400；DEV 仅诊断。','',
           '|种子|当前事件探针|事件|FIT 平衡准确率|DEV 平衡准确率|','|---|---|---|---:|---:|']
    def fmt(x):return '不可计算' if x is None else f'{100*x:.1f}%'
    for seed in cfg['seeds']:
        for kind in cfg['probe_kinds']:
            for target in ('current_anchor_SEE2','current_terminal_SEE2'):
                z={r['split']:r['balanced_accuracy'] for r in event if r['seed']==seed and r['kind']==kind and r['target']==target and r['house'] is None}
                lines.append(f"|{seed}|{kind}|{target}|{fmt(z['FIT'])}|{fmt(z['DEV'])}|")
    lines+=['','|种子|冻结记忆来源|FIT 历史状态探针|DEV 历史状态探针|','|---|---|---:|---:|']
    for seed in cfg['seeds']:
        for arm in cfg['memory_probes']:
            z={r['split']:r['balanced_accuracy'] for r in memory if r['seed']==seed and r['arm']==arm and r['house'] is None}
            lines.append(f"|{seed}|{arm}|{fmt(z['FIT'])}|{fmt(z['DEV'])}|")
    lines+=['','|种子|B2 原状态头：DEV 历史状态平衡准确率|当前终点状态平衡准确率|','|---|---:|---:|']
    for seed in cfg['seeds']:
        z={r['bit']:r['balanced_accuracy'] for r in state if r['seed']==seed and r['arm']=='B2' and r['split']=='DEV' and r['task']=='task_A' and r['phase']=='all'}
        lines.append(f"|{seed}|{fmt(z[1])}|{fmt(z[2])}|")
    lines+=['','## 梯度与动作','']
    for name in ('action_vs_auxiliary','action_vs_preservation'):
        z=[r['cosines'][name] for r in gradient if r['cosines'][name] is not None]
        lines.append(f"- {name}：负余弦 {sum(x<0 for x in z)}/{len(z)}；中位数 {statistics.median(z):.4f}。仅为固定 FIT 样本上的局部梯度关系。")
    for seed in cfg['seeds']:
        z=[r for r in actions if r['seed']==seed and r['arm']=='B2' and r['split']=='DEV' and r['task']=='task_A' and r['state']=='state_correct'][0]
        lines.append(f"- seed {seed}：B2 四位状态都预测正确的 DEV 教师决策 {z['n']} 条，与已登记教师动作集合一致 {z['any_registered_teacher_agreement']} 条。")
    lines+=['','## 判断边界','']+['- '+x for x in result['limitations']]
    with (OUT/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':main()
