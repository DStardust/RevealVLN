"""Matched new versus old FIRST policies on the fixed exposed unseen block."""
import csv
import importlib.util
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
import common as u
spec=importlib.util.spec_from_file_location('scope_review',HERE.parent/'scope_validation_v1/review.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
records=old.records


def summary(run):
    if not (run/'PROTOCOL.json').exists():
        return dict(status='AWAITING_TRAINING',complete_groups=0,planned_groups=80,planned_executions=1040)
    r=old.summary(run)
    r.update(scope='FIRST_ALIGNED_VERSUS_ALL_TRAINED_WITH_SAME_FIRST_INFERENCE',
        optimizer_updates_in_training=18000,training_not_navigation_proof=True)
    return r


def review(run):
    u.verify_sources(run);p=u.read(run/'PROTOCOL.json');r=summary(run);rows=records(run)
    if not r['evaluation_complete']:raise ValueError('INCOMPLETE_UNSEEN_DENOMINATOR')
    table=[]
    for i,row in sorted(rows.items()):
        for arm,out in row['outcomes'].items():
            path=Path(row['path']).parent/arm/'TRACE.jsonl'
            if u.sha(path)!=row['trace_hashes'][arm]:raise ValueError('TRACE_CHANGED')
            table.append(dict(id=i,house=row['house'],arm=arm,success=out['success'],spl=out['spl'],
                steps=out['steps'],budget_penalty=out['steps']/500 if out['success'] else 1.,trace=str(path),trace_sha256=row['trace_hashes'][arm]))
    with (run/'ROLLOUTS.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    comparisons={}
    for mode in ('CURRENT','DELTA'):
        by_seed={s:r['paired'][f'ALIGNED_minus_OLD_{mode}_s{s}'] for s in p['seeds']}
        comparisons[mode]=dict(by_seed=by_seed,mean_delta_sr=statistics.mean(v['delta_sr'] for v in by_seed.values()),
            positive_seeds=sum(v['delta_sr']>0 for v in by_seed.values()),
            mean_delta_spl=statistics.mean(r['arms'][f'ALIGNED_{mode}_FIRST_s{s}']['spl']-r['arms'][f'OLD_{mode}_FIRST_s{s}']['spl'] for s in p['seeds']),
            mean_delta_steps=statistics.mean(r['arms'][f'ALIGNED_{mode}_FIRST_s{s}']['mean_steps']-r['arms'][f'OLD_{mode}_FIRST_s{s}']['mean_steps'] for s in p['seeds']))
    r.update(alignment_effect=comparisons,automatic_adoption=False,
        interpretation='Actual optimization attempt. Shared supervision change cannot be attributed uniquely to DELTA. Exposed unseen80 is development evidence.')
    u.write(run/'RESULT.json',r)
    lines=['COMPLETE','',f"固定80条unseen，{len(table)}/1040次执行。旧、新FIRST与原生均在本轮同进程重新运行。",'']
    for mode,v in comparisons.items():lines.append(f"{mode}：对齐训练−旧训练平均ΔSR={v['mean_delta_sr']:.4f}，ΔSPL={v['mean_delta_spl']:.4f}，Δ动作={v['mean_delta_steps']:.2f}；正向种子{v['positive_seeds']}/3。")
    lines+=['','首次位置监督是两种架构共有的训练修订，不能单独归为DELTA架构贡献。','80条已反复暴露，属于开发证据，不是新盲测、完整1839或自动部署。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return r
