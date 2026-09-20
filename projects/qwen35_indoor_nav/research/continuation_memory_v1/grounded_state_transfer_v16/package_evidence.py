"""Local review index of actual artifacts, including failures and missing denominators."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *

def main():
    runs=[];resources=[];families={};rollouts=0;models=0;golden=[];formal=[]
    for run in sorted((HERE/'runs').iterdir()):
        if not run.is_dir():continue
        item=dict(run=run.name)
        for file in ('PILOT_RESULT.json','DATA_SHORTFALL.json','STATUS.json'):
            if (run/file).exists():item[file]=read(run/file)
        for p in run.glob('HOUSE_*.json'):
            for family in read(p)['families']:families[family['family_id']]=family
        if (run/'RESOURCE_SESSIONS.jsonl').exists():
            resources.extend(dict(run=run.name,**r) for r in c.records(run/'RESOURCE_SESSIONS.jsonl'))
        for p in run.glob('numerical_pilot/RESULT.json'):golden.append(dict(run=run.name,**read(p)))
        if (run/'EVALUATION_REGISTRY.json').exists():
            from evaluate_continuations import admitted
            formal.append(dict(run=run.name,completed_models=len(list(run.glob('train/*/RESULT.json'))),
                admitted_rollouts=9*len(admitted(run,read(run/'EVALUATION_REGISTRY.json'))),
                physical_rollout_files=len(list(run.glob('evaluate/session_*/rollouts/*/ROLLOUT.json')))))
        runs.append(item)
    by_split={s:sum(f['split']==s for f in families.values()) for s in ('FIT','DEV','TEST')}
    houses={s:sorted({f['house'] for f in families.values() if f['split']==s}) for s in by_split}
    jobs=[]
    for p in (HERE/'standalone_jobs').glob('*/STATUS.json'):jobs.append(dict(job=p.parent.name,**read(p)))
    active=[r['job'] for r in jobs if r['status']=='RUNNING']
    if formal:models=formal[-1]['completed_models'];rollouts=formal[-1]['admitted_rollouts']
    summary=dict(status='RUNNING' if active else 'REGISTERED_EXPERIMENT_INCOMPLETE' if rollouts<720 else 'SEE_RUN_REVIEW',snapshot_unix=time.time(),formal_runs=formal,
        baseline_commit=BASELINE,unique_certified_families=len(families),families_by_split=by_split,houses=houses,
        planned_families=26,planned_models=9,completed_models=models,planned_continuations=720,physical_rollout_files=rollouts,
        method_effect='UNKNOWN' if rollouts<720 else 'SEE_COMPLETE_PAIRED_REVIEW',adopted=False,
        gpu_session_hours=sum(r['wall_seconds'] for r in resources)/3600,active_jobs=active,
        resources=resources,golden_pilots=golden,runs=runs,
        old_v15_modified=False,scientific_claim='No Ours increment has been established by generator/numerical pilots.')
    write(HERE/'RESULT.json',summary)
    write(HERE/'SOURCE_LOCK.json',source_lock())
    latest=golden[-1] if golden else None
    text=f"# {summary['status']}\n\nV16 训练模型 {models}/9，自主续接 {rollouts}/720；主任务计划每臂 N=192，尚无完整效应比较。干预获益 UNKNOWN，未采用。\n\n"
    text+=f"实际已认证族 {len(families)}/26：FIT={by_split['FIT']}，DEV={by_split['DEV']}，TEST={by_split['TEST']}。去重后统计，跨 run 复用不重复计数。屋清单：`{json.dumps(houses,ensure_ascii=False)}`。\n\n"
    if latest:text+=f"冻结 best4k 已加载；FIT 黄金输入 {latest['golden_inputs']} 条，真实前向 {latest['real_qwen_forwards']} 次，原生 logits 最大差 {latest['max_logit_delta']}，argmax 翻转 {latest['native_argmax_flips']}；基座首尾未变。这是数值/运行证据，不是方法收益。\n\n"
    text+=f"已结束 GPU 会话累计 {summary['gpu_session_hours']:.4f} 小时，包含失败。运行中的服务：`{active}`；活跃会话消耗未加入已结束账本，见对应 RESOURCES.jsonl。\n\n"
    text+="共同修复包括统一 method argmax、前瞻碰撞安全终点、稠密因果监督、可恢复 optimizer/RNG 与完整组封存；不能将这些公共变化归于 Ours。现有 V15 18 模型、216 条续接、87 个 UNKNOWN 及旧 DATA 准入标记均未修改。\n\n"
    text+="已实际读取固定入口、V15 报告/源码锁/资产清单、数据/teacher/objective/训练/编码器/runtime_r2/旧 STOP 选择器与原 SEE2 编译器。可用本地资产包括 best4k、底模、项目 Python、Qwen/Habitat 环境、MP3D 场景和原始数组。初次路径猜测 feedback_v12/feedback.py 与 runtime_r2/prefix_replay.py 不存在；实际实现分别位于 feedback_generation_v1/feedback.py 与 runtime_r2/continuation_service.py，已改为读取真实入口。\n\n"
    text+="早期采集失败和 CuBLAS 环境缺项保留在 runs/。同一族中的多个任务、续接和相同物理历史不计作独立样本。SEE2 始终是同一实例连续两帧各至少256像素，不等于到访房间。没有完整 R2R SR、SR40、新架构或真机部署证据。\n\n"
    text+="运行命令见 README.md；独立 service 的 PPID=1、UID/GID、cgroup、退出码见 standalone_jobs。pipeline 按同一注册协议自行推进，未达完整数据规模时明确报不足，不把部分数据当正式720条实验。\n"
    (HERE/'REPORT_ZH.md').write_text(text)
    entries=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(HERE)
        if any(x in rel.parts for x in ('cache','tmp','__pycache__','source','content')):continue
        if p.name in ('EVIDENCE_MANIFEST.json','EVIDENCE_LOGS.tar.gz') or '.tmp.' in p.name:continue
        if p.suffix not in ('.py','.json','.jsonl','.md','.csv','.txt','.log'):continue
        if any(active_job in str(rel) for active_job in active):continue
        entries.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'EVIDENCE_MANIFEST.json',dict(entries=entries,scope='Local reviewable code and logs; raw arrays/checkpoints/caches stay at referenced paths, not silently omitted from data counts.',active_jobs=active))
    print(json.dumps({k:summary[k] for k in ('status','unique_certified_families','completed_models','physical_rollout_files','gpu_session_hours','active_jobs')},ensure_ascii=False))

if __name__=='__main__':main()
