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
    text=f"# 普通导航 STOP 行训练与闭环检验\n\n{result['status']}，完整 {result['complete_pairs']}/100 对。\n\n本轮原生 A 成功 {a['successes']}，候选 B 成功 {b['successes']}；新增成功 {len(result['wins'])}，丢失成功 {len(result['losses'])}。完整 ΔSR：{result['delta_sr']}。缺项不从计划分母删除。\n\n仅 STOP 行在 16 个 FIT 屋的真实状态上重新训练；运动输出行、Qwen/LoRA、观察与评测不变。12/4 屋离线诊断不等于闭环收益；缓存跨会话数值差异见各 CACHE_LIVE_PARITY.json，旧 V12 FAIL 未改写。\n\n当前不自动部署；完整 100 对才判断开发信号。没有独立测试泛化、完整 val_unseen SR 或新算法贡献。若失败，不改成功定义、参数或挑选重跑。\n\n逐对轨迹在 sessions/；训练记录 TRAIN_RESULT.json；完整机器可读表 REVIEW.json。\n"
    (run/'REPORT_ZH.md').write_text(text)
    return result
if __name__=='__main__':main(Path(sys.argv[1]))
