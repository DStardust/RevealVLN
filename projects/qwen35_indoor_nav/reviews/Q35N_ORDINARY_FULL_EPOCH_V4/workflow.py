"""Reuse bounded supervision; extend only to the declared one-epoch endpoint."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'Q35N_ORDINARY_CONTINUE_V2/workflow.py'
raw=OLD.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e437bfbbf67f4455db00255e75e6577da8aec1c70ce64704efba063735a14e55'
text=raw.decode()
changes={
 'ordinary_expanded_continue_v2':'ordinary_expanded_full_epoch_v4',
 'ordinary_expanded_continue_dev_v2':'ordinary_full_epoch_dev_v4',
 'q35n_expanded_continue_v2':'q35n_full_epoch_v4',
 'time.monotonic()-start<4800':'time.monotonic()-start<15000',
 'checkpoint_000004200.pt.json':'checkpoint_000008200.pt.json',
 "final['cursor']['updates']==8000 and final['stop']==['BUDGET:max_updates']":"final['cursor']==read(TRAIN/'PROTOCOL_FILESTORE.json')['expected_final_cursor'] and final['status']=='EPOCHS_COMPLETED' and final['stop']==[]",
 "        status('BINDING_FINAL_CHECKPOINT')":"""        status('VERIFYING_FULL_EPOCH_CHECKPOINT')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        run_logged([str(QPY),'-I','-B',str(HERE/'accept_final.py')],'final_acceptance.log',300,env)
        status('BINDING_FINAL_CHECKPOINT')""",
 'EVALUATING_FINAL_8000':'EVALUATING_FINAL_31059',
}
for old,new in changes.items():
    assert old in text,old;text=text.replace(old,new)

CLOSE_REPORT='''def close_report():
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    s=importlib.util.spec_from_file_location('unchanged_full_epoch_analyzer',ANALYZER)
    a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    before=a.case_report(BASE);after=a.case_report(CASE);paired=a.paired(before,after)
    assert paired['matched_inference_batch']
    delta=paired['delta'];positive=delta['success']>0 and delta['spl']>=0 and delta['ndtw']>=-.01
    result=dict(status='COMPLETE',unix=time.time(),before=before,after=after,paired=paired,
        positive_development_signal=positive,scientific_gain_verified=False,controller_enabled=False,
        automatic_next_training=False,checkpoint_selection='fixed epoch1, position0, updates31059',
        source_and_input_locks_unchanged=True,unique_epoch_decisions=2650347,new_decisions=1967696,
        low_lr_failed_extra_compute_decisions=341939)
    save(HERE/'RESULT.json',result)
    report='# 普通导航：完整一轮覆盖结果\\n\\n同100条、5屋INTERNAL_DEV，batch=1、纯模型、500动作。不代表独立论文确认。\\n\\n'
    report+='| 版本 | SR | SPL | nDTW | OSR |\\n|---|---:|---:|---:|---:|\\n'
    for label,case in [('较好4000步（13%覆盖）',before),('固定31059步（完整一轮）',after)]:
        r=case['result'];report+=f"| {label} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {100*r['osr']:.2f}% |\\n"
    report+=f"\\n新增成功{int(paired['wins'])}条、失去{int(paired['losses'])}条。事前工程门槛：{'满足' if positive else '不满足'}。\\n\\n"
    report+='| 房屋 | ΔSR（百分点） | ΔSPL | ΔnDTW |\\n|---|---:|---:|---:|\\n'
    for house,d in paired['by_house_delta'].items():report+=f"| {house} | {100*d['success']:+.2f} | {100*d['spl']:+.2f} | {100*d['ndtw']:+.2f} |\\n"
    report+='\\n完整一轮2,650,347个唯一计划决策，无丢弃尾批；本段新增1,967,696。保留原8,000步的优化器/游标/5e-5峰值余弦调度，只补完原顺序。低LR失败分支另付341,939个决策计算，不计新覆盖。没有STOP偏置、视觉停滞保护或特殊数据，仍不是UAD任务记忆架构创新。\\n\\n'
    report+='历史结果保留：高LR8,000步SR14%、低LR8,000步13%、FIT停止校准真实DEV16%，均未采用。较好的4,000步权重不覆盖。逐屋、留一屋与描述性五屋bootstrap在RESULT.json；无自动第二轮，未独立确认。\\n'
    with (HERE/'REPORT_ZH.md').open('x') as f:f.write(report)
    return result


'''
start=text.index('def close_report():');end=text.index('def main():',start)
text=text[:start]+CLOSE_REPORT+text[end:]
exec(compile(text,str(HERE/'workflow.py')+':bounded-parent','exec'),globals())
