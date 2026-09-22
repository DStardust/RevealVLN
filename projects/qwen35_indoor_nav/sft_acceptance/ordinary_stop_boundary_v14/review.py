"""Recompute complete-pair denominators, expose missing pairs and STOP tradeoffs."""
import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
def summarize(run,verify=False):
    found={}
    for p in run.glob('sessions/*/pairs/*/PAIR.json'):
        row=u.read(p)
        if row['rank'] in found:raise ValueError('DUPLICATE_PAIR')
        if not row['state_unchanged'] or not row['input_prefix_matched']:raise ValueError('INVALID_SEAL')
        if verify:
            for f,h in row['trace_hashes'].items():
                if u.sha(p.parent/f)!=h:raise ValueError('TRACE_CHANGED')
            index=u.read(u.V5/'PAIR_ORDER.json')[row['rank']]['index']
            for arm in ('A','B'):u.c.audit_episode(p.parent/arm,index)
        found[row['rank']]=row
    rows=list(found.values());n=len(rows);arms={}
    for arm in ('A','B'):
        rr=[x[arm] for x in rows];p=sum(x['success'] for x in rr)
        arms[arm]=dict(successes=p,complete=n,planned=100,partial_sr=p/n if n else None,
            full_sr=p/100 if n==100 else None,missing=100-n,identification_bounds=[p/100,(p+100-n)/100],
            spl=sum(x['spl'] for x in rr)/n if n else None,ndtw=sum(x['ndtw'] for x in rr)/n if n else None,
            oracle_success=sum(x['oracle_success'] for x in rr)/n if n else None,
            decisions=sum(x['steps'] for x in rr),collisions=sum(x['collisions'] for x in rr),
            failures=dict(collections.Counter(x['failure_category'] for x in rr)))
    wins=[x['episode_id'] for x in rows if x['B']['success']>x['A']['success']]
    losses=[x['episode_id'] for x in rows if x['B']['success']<x['A']['success']]
    houses=[]
    for house in sorted({x['house'] for x in rows}):
        r=[x for x in rows if x['house']==house];a=sum(x['A']['success'] for x in r);b=sum(x['B']['success'] for x in r)
        houses.append(dict(house=house,complete=len(r),A=a,B=b,delta_sr=(b-a)/len(r)))
    status='VALID_COMPLETE' if n==100 else 'PARTIAL'
    return dict(status=status,complete_pairs=n,planned_pairs=100,missing_ranks=sorted(set(range(100))-set(found)),arms=arms,wins=wins,losses=losses,
        retained_successes=sum(bool(x['A']['success'] and x['B']['success']) for x in rows),houses=houses,
        delta_sr=(len(wins)-len(losses))/100 if n==100 else None,partial_delta_sr=(len(wins)-len(losses))/n if n else None,
        max_base_logit_delta=max((x['max_base_logit_delta'] for x in rows),default=None),
        logits_bitwise_equal=all(x['logits_bitwise_equal'] for x in rows) if n else None,
        adoption='NOT_AUTOMATICALLY_ADOPTED',scope='Repeatedly exposed INTERNAL_DEV100, five houses; no independent generalization claim')
def main(run):
    result=summarize(run,True);u.write(run/'REVIEW.json',result)
    a,b=result['arms']['A'],result['arms']['B']
    text=f"# 普通导航停车边界修复\n\n{result['status']}，完整 {result['complete_pairs']}/100 对。\n\nA 是已保留的 V13 停车头（上轮25%，本轮重新实际运行），B 是 V14 同轨迹边界排序监督。A 成功 {a['successes']}，B 成功 {b['successes']}；新增 {len(result['wins'])}，丢失 {len(result['losses'])}。完整 ΔSR：{result['delta_sr']}。缺项不从计划分母删除。\n\n使用既有5487个真实输入，派生888对边界监督；没有新物理轨迹。距离、配对和标签仅进入离线loss。只改变STOP训练目标；运行期仍为原四类argmax，输入/运动输出/Qwen/LoRA/3m主动STOP定义不变。\n\n12/4屋只作离线诊断；无参数搜索。最终全16屋训练的固定候选接受完整闭环检验。已暴露的五屋INTERNAL_DEV，不是独立泛化或完整val_unseen结论。无自动部署。\n\n日志在sessions，训练证据TRAIN_RESULT.json和OPTIMIZER.jsonl，具体物理样本证书DATA_MANIFEST.json。\n"
    (run/'REPORT_ZH.md').write_text(text)
    return result
if __name__=='__main__':main(Path(sys.argv[1]))
