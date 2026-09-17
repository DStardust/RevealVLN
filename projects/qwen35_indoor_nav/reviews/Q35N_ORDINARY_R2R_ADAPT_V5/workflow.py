"""Reuse bounded, exact-identity workflow with one fixed R2R endpoint."""
import hashlib
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'Q35N_ORDINARY_CONTINUE_V2/workflow.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='e437bfbbf67f4455db00255e75e6577da8aec1c70ce64704efba063735a14e55'
text=PARENT.read_text()
changes={
"ordinary_expanded_continue_v2":"ordinary_r2r_adapt_v5",
"ordinary_expanded_continue_dev_v2":"ordinary_r2r_adapt_dev_v5",
"q35n_expanded_continue_v2":"q35n_r2r_adapt_v5",
"        status('BINDING_FINAL_CHECKPOINT')":"        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')\n        run_logged([str(QPY),'-I','-B',str(HERE/'accept_final.py')],'final_acceptance.log',300,env)\n        status('BINDING_FINAL_CHECKPOINT')",
}
for old,new in changes.items():
    assert old in text,old
    text=text.replace(old,new)
tail="if __name__=='__main__':"
assert text.count(tail)==1
exec(compile(text[:text.index(tail)],__file__+':hash-bound-parent','exec'),globals())

def positive_gate(delta):
    return delta['success']>0 and delta['spl']>=0 and delta['ndtw']>=-.01

def close_report():
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    s=importlib.util.spec_from_file_location('r2r_unchanged_analyzer',ANALYZER)
    a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    before=a.case_report(BASE);after=a.case_report(CASE);paired=a.paired(before,after)
    assert paired['matched_inference_batch']
    control=a.case_report(LINE/'closed_loop_bench/ordinary_expanded_continue_dev_v2')
    source_comparison=a.paired(control,after)
    positive=positive_gate(paired['delta'])
    result=dict(status='COMPLETE',unix=time.time(),before=before,after=after,paired=paired,
        source_control_comparison=source_comparison,positive_development_signal=positive,
        scientific_gain_verified=False,controller_enabled=False,automatic_next_training=False,
        checkpoint_selection='fixed R2R8000; no intermediate selection',source_and_input_locks_unchanged=True,
        new_decisions=396001,branch_decisions=736713)
    save(HERE/'RESULT.json',result)
    text='# 普通导航：R2R来源适配结果\n\n同100条、5个已暴露INTERNAL_DEV屋，batch1、纯模型、500动作。不是独立确认。\n\n'
    text+='| 版本 | SR | SPL | nDTW | OSR |\n|---|---:|---:|---:|---:|\n'
    for label,case in [('较好4000步',before),('混合来源8000步对照',control),('R2R来源8000步',after)]:
        r=case['result'];text+=f"| {label} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {100*r['osr']:.2f}% |\n"
    text+=f"\n相对较好4000步新增成功{paired['wins']}条、失去{paired['losses']}条；事前工程门槛：{'满足' if positive else '不满足'}。\n\n"
    text+='| 房屋 | ΔSR（百分点） | ΔSPL | ΔnDTW |\n|---|---:|---:|---:|\n'
    for house,d in paired['by_house_delta'].items():text+=f"| {house} | {100*d['success']:+.2f} | {100*d['spl']:+.2f} | {100*d['ndtw']:+.2f} |\n"
    text+='\n保留4000步参数、AdamW矩、原LR时钟和模型，只把新增4000更新改为R2R人工指令来源，新增396,001个样本，与本扩产分支前4000步无索引重复。训练步数和token上限相同，但来源文本长度导致实际决策数不同，不宣称精确动作等预算。\n'
    text+='\n完整一轮失败（SR14%）、低LR失败（13%）、STOP校准失败（16%）与所有历史结果保留。开发集已多次用于工程决策，存在适应性选择风险；五屋bootstrap和留一屋仅描述，见RESULT.json。任何正号都不是UAD架构创新或论文最终收益，特殊数据仍未混入。\n'
    with (HERE/'REPORT_ZH.md').open('x') as f:f.write(text)
    return result

if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['run'])
    freeze() if sys.argv[1]=='freeze' else main()
