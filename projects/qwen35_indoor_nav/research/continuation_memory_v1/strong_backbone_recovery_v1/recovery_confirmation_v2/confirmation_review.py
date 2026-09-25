"""Full denominators; paired seeds are not independent houses or episodes."""
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as u
from transfer_pipeline import records


def summarize(rows,entries,p):
    n=len(entries);names=['NATIVE']+list(p['models']);arms={};paired={}
    assert set(rows)<=set(e['id'] for e in entries)
    for row in rows.values():assert set(row['outcomes'])==set(names),'MISSING_MODEL_IN_GROUP'
    for arm in names:
        values=[r['outcomes'][arm] for r in rows.values()];s=sum(v['success'] for v in values)
        arms[arm]=dict(successes=s,complete=len(values),planned=n,sr=s/n if len(values)==n else None,
            identification_bounds=[s/n,(s+n-len(values))/n],
            spl=statistics.mean(v['spl'] for v in values) if values else None,
            mean_steps=statistics.mean(v['steps'] for v in values) if values else None)
    comparisons=[(name,'NATIVE') for name in p['models']]
    comparisons += [(f'CONCAT_s{s}',f'LOCAL_s{s}') for s in p['seeds']]
    for arm,reference in comparisons:
        wins=[i for i,r in rows.items() if r['outcomes'][arm]['success']>r['outcomes'][reference]['success']]
        losses=[i for i,r in rows.items() if r['outcomes'][arm]['success']<r['outcomes'][reference]['success']]
        paired[arm+'_vs_'+reference]=dict(wins=wins,losses=losses,
            retained_successes=sum(r['outcomes'][arm]['success'] and r['outcomes'][reference]['success'] for r in rows.values()),
            delta_sr=(len(wins)-len(losses))/n if len(rows)==n else None,
            difference_identification_bounds=[(len(wins)-len(losses)-(n-len(rows)))/n,(len(wins)-len(losses)+(n-len(rows)))/n])
    kinds={e['id']:e.get('kind','NATURAL') for e in entries}
    by_kind={k:{a:dict(planned=sum(v==k for v in kinds.values()),complete=sum(kinds[i]==k for i in rows),
        successes=sum(r['outcomes'][a]['success'] for i,r in rows.items() if kinds[i]==k)) for a in names} for k in sorted(set(kinds.values()))}
    by_house={h:{a:dict(planned=sum(e['house']==h for e in entries),complete=sum(r['house']==h for r in rows.values()),
        successes=sum(r['outcomes'][a]['success'] for r in rows.values() if r['house']==h)) for a in names} for h in sorted({e['house'] for e in entries})}
    seed_summary={}
    if len(rows)==n:
        for architecture in ('CONCAT','LOCAL'):
            sr=[arms[f'{architecture}_s{s}']['sr'] for s in p['seeds']]
            seed_summary[architecture]=dict(mean_sr=statistics.mean(sr),seed_range=[min(sr),max(sr)],
                deltas_vs_native=[v-arms['NATIVE']['sr'] for v in sr])
        seed_summary['history_deltas']=[paired[f'CONCAT_s{s}_vs_LOCAL_s{s}']['delta_sr'] for s in p['seeds']]
    return dict(complete_groups=len(rows),planned_groups=n,evaluation_complete=len(rows)==n,arms=arms,paired=paired,
        by_kind=by_kind,by_house=by_house,seed_summary=seed_summary,split=p.get('split'),
        uncertainty_scope='Seed range is descriptive; seeds share episodes and houses. Identification bounds are not confidence intervals.')


def summary(run):
    p=u.read(run/'PROTOCOL.json');rows=records(run,'evaluation',True)
    return summarize(rows,u.read(run/'DATA_MANIFEST.json')['episodes'],p)


def final_report(run,hours):
    p=u.read(run/'PROTOCOL.json');root=run/'unseen'
    for session in (root/'evaluation').glob('*'):
        seal=u.read(session/'STATE_SEAL.json')
        assert seal['heads_unchanged'] and seal['base_before']==seal['base_after']==p['expected_base_state_sha256']
        for path in session.glob('episodes/*/COMPLETE.json'):
            group=u.read(path)
            assert u.sha(session/'RUNTIME_IDENTITY.json')==group['runtime_identity_sha256']
            for arm,h in group['trace_hashes'].items():assert u.sha(path.parent/arm/'TRACE.jsonl')==h
            assert len(group['audits'])==len(p['models'])
            assert len(group['seed_pair_audits'])==len(p['seeds'])
            assert all(a['input_prefix_matched'] and a['action_prefix_matched'] and a['argmax_flip_count']==0 for a in group['seed_pair_audits'].values())
            assert all(a['input_prefix_matched'] and a['action_prefix_matched'] and a['argmax_flip_count']==0 for a in group['audits'].values())
    result=summary(root);assert result['evaluation_complete']
    result.update(status='COMPLETE',gpu_hours=hours,base_updates=0,adopted=False,full_1839=False,exposed_unseen=True)
    u.write(run/'RESULT.json',result)
    lines=['COMPLETE','','固定已暴露 unseen200，三种子、同数据同预算；LOCAL 保留基座历史，新增分支只编码当前观测；使用单位写入增益避免人为压低该对照。','']
    for name,value in result['arms'].items():lines.append(f"{name}: {value['successes']}/200，SR {value['sr']:.3f}，SPL {value['spl']:.4f}")
    lines+=['','各 seed 的 CONCAT−LOCAL：'+str(result['seed_summary']['history_deltas']),
        '各 seed 的 CONCAT−NATIVE：'+str(result['seed_summary']['CONCAT']['deltas_vs_native']),
        '种子共用同一批路线，不能当成 600 个独立泛化样本；没有自动采用或论文新颖性结论。',
        '这项对照衡量新增递归历史的作用，不分离恢复数据与保护损失各自的因果贡献。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
