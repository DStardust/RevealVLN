"""Thirteen-arm complete-group scope comparison; never recycle old outcomes."""
import csv
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import common as u


def records(run):
    if not (run/'PROTOCOL.json').exists(): return {}
    p=u.read(run/'PROTOCOL.json'); names={'NATIVE',*p['heads']}; rows={}
    for session in sorted((run/'evaluation').glob('*')):
        if not (session/'STATE_SEAL.json').exists(): continue
        seal=u.read(session/'STATE_SEAL.json'); identity=u.read(session/'RUNTIME_IDENTITY.json')
        if seal['base_before']!=seal['base_after'] or seal['base_before']!=p['expected_base_state_sha256'] or not seal['heads_unchanged']:
            raise ValueError('INVALID_STATE_SEAL')
        if identity['heads']!=p['heads'] or identity['base_updates'] or identity['seed_pairs']!=p['comparisons']:
            raise ValueError('RUNTIME_REGISTRY_CHANGED')
        identity_sha=u.sha(session/'RUNTIME_IDENTITY.json')
        for path in session.glob('episodes/*/COMPLETE.json'):
            r=u.read(path)
            if r['id'] in rows or set(r['outcomes'])!=names or r['runtime_identity_sha256']!=identity_sha:
                raise ValueError('INCOMPLETE_OR_CHANGED_GROUP')
            if set(r['audits'])!=set(p['heads']) or set(r['seed_pair_audits'])!=set(p['comparisons']):
                raise ValueError('MISSING_REGISTERED_PREFIX_AUDIT')
            for audit in [*r['audits'].values(),*r['seed_pair_audits'].values()]:
                if not audit['input_prefix_matched'] or not audit['action_prefix_matched'] or audit['argmax_flip_count']:
                    raise ValueError('INVALID_PAIR_PREFIX')
            rows[r['id']]=dict(r,path=str(path))
    return rows


def summarize(p, entries, rows):
    n=len(entries); ids={e['id'] for e in entries}; names=['NATIVE',*p['heads']]
    if not n or len(ids)!=n or not set(rows)<=ids: raise ValueError('DENOMINATOR_CHANGED')
    if any(set(r['outcomes'])!=set(names) for r in rows.values()): raise ValueError('INCOMPLETE_GROUP')
    complete=len(rows)==n; arms={}
    for arm in names:
        values=[r['outcomes'][arm] for r in rows.values()]; wins=sum(v['success'] for v in values)
        arms[arm]=dict(successes=wins,complete=len(values),planned=n,sr=wins/n if complete else None,
            sr_identification_bounds=[wins/n,(wins+n-len(values))/n],
            spl=statistics.mean(v['spl'] for v in values) if complete else None,
            mean_steps=statistics.mean(v['steps'] for v in values) if complete else None)
    pairs=dict(p['comparisons'])
    pairs.update({a+'_vs_NATIVE':{'CURRENT':'NATIVE','DELTA':a} for a in p['heads']})
    comparisons={}
    for name,pair in pairs.items():
        left,right=pair['CURRENT'],pair['DELTA']
        win=[i for i,r in rows.items() if r['outcomes'][right]['success']>r['outcomes'][left]['success']]
        loss=[i for i,r in rows.items() if r['outcomes'][right]['success']<r['outcomes'][left]['success']]
        net=len(win)-len(loss)
        comparisons[name]=dict(reference=left,candidate=right,wins=win,losses=loss,
            delta_sr=net/n if complete else None,
            difference_identification_bounds=[(net-(n-len(rows)))/n,(net+(n-len(rows)))/n])
    by_house={h:{a:dict(planned=sum(e['house']==h for e in entries),
        complete=sum(r['house']==h for r in rows.values()),
        successes=sum(r['outcomes'][a]['success'] for r in rows.values() if r['house']==h)) for a in names}
        for h in sorted({e['house'] for e in entries})}
    return dict(status='COMPLETE' if complete else 'PARTIAL',evaluation_complete=complete,
        planned_groups=n,complete_groups=len(rows),planned_executions=n*len(names),
        arms=arms,paired=comparisons,by_house=by_house,split='val_unseen',exposed=True,full_1839=False,
        old_candidate_retained=True,automatic_replacement=False,optimizer_updates=0,
        scope='Same frozen head; FIRST minus ALL. Not a training or novel architecture gain.',
        statistical_unit='Shared houses/routes/seeds; executions are not independent generalization samples.')


def summary(run):
    return summarize(u.read(run/'PROTOCOL.json'),u.read(run/'DATA_MANIFEST.json')['episodes'],records(run))


def review(run):
    u.verify_sources(run); rows=records(run); p=u.read(run/'PROTOCOL.json')
    result=summarize(p,u.read(run/'DATA_MANIFEST.json')['episodes'],rows)
    if not result['evaluation_complete']: raise ValueError('INCOMPLETE_PLANNED_EVALUATION')
    table=[]
    for i,r in sorted(rows.items()):
        for arm,outcome in r['outcomes'].items():
            trace=Path(r['path']).parent/arm/'TRACE.jsonl'
            if u.sha(trace)!=r['trace_hashes'][arm]: raise ValueError('TRACE_CHANGED')
            table.append(dict(id=i,house=r['house'],arm=arm,success=outcome['success'],spl=outcome['spl'],
                steps=outcome['steps'],trace=str(trace),trace_sha256=r['trace_hashes'][arm]))
    with (run/'ROLLOUTS.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    u.write(run/'RESULT.json',result)
    lines=['COMPLETE','',f'{len(rows)} 条固定 unseen，{len(table)} 次执行；0 次训练更新。',
        '本轮只检验减少干预位置是否改善自主执行。复用同一权重，同进程重新运行 ALL 对照；不混用旧轨迹。','']
    for arm,v in result['arms'].items(): lines.append(f"{arm}: SR={v['sr']:.4f}, SPL={v['spl']:.4f}, steps={v['mean_steps']:.2f}")
    lines+=['','已暴露开发诊断，不是完整1839或独立泛化。旧CONCAT候选保留，不自动采用新版本。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return result
