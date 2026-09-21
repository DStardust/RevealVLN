"""Finish CPU readiness independently; deliberately stops before GPU training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *


def main(run):
    cfg=read(HERE/'PROTOCOL.json');began=time.monotonic()
    if (run/'RESULT.json').exists():
        verify_binding(run)
        print(json.dumps(read(run/'RESULT.json')),flush=True)
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def execute(name,args):
        with (run/(name+'.log')).open('x') as stream:
            subprocess.run([cfg['torch_python'],'-I','-B',*args],cwd=HERE,env=env,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not run.exists():
        subprocess.run([cfg['torch_python'],'-I','-B',str(HERE/'prepare.py'),'--run',str(run)],env=env,check=True)
    try:
        if not (run/'CPU_TEST_RESULT.json').exists():
            execute('cpu_contracts',[str(HERE/'test_cpu.py'),'--run',str(run)])
        tests=read(run/'CPU_TEST_RESULT.json')
        if tests['status']!='PASS' or tests['cuda_initialized']:
            raise ValueError('CPU_CONTRACT_FAILURE')
        verify_binding(run)
        records=[]
        for mode in cfg['modes']:
            write(run/'STATUS.json',dict(status='CPU_SMOKE_RUNNING',mode=mode,gpu_started=False))
            target=run/'cpu_smoke'/mode
            if not (target/'RESULT.json').exists():
                execute('train_'+mode,[str(HERE/'train.py'),'--run',str(run),'--output',str(target),
                                      '--mode',mode,'--device','cpu','--updates',str(cfg['cpu_smoke_updates'])])
            record=read(target/'RESULT.json')
            if record['cuda_initialized'] or record['base_updates'] or not record['parameters_changed']:
                raise ValueError('CPU_SMOKE_NOT_AS_REGISTERED')
            records.append(record)
        result=dict(status='READY_FOR_GPU_STAGE_NOT_LAUNCHED',cpu_tests=tests['tests'],smoke=records,
                    smoke_optimizer_updates=sum(r['updates'] for r in records),isolated_unit_test_optimizer_updates=12,
                    gpu_hours=0,new_base_forwards=0,new_physical_executions=0,navigation_gain=None,
                    method_contribution='UNTESTED',cpu_driver_seconds=time.monotonic()-began,
                    stop_reason='User requested progress up to the GPU boundary; existing KL test remains independent.',
                    next='Fresh GPU training from shared initialization, then runtime integration/cache-live audit before any navigation claim. CPU smoke weights are not training initializations.')
        write(run/'RESULT.json',result,True)
        evidence=tests['evidence'];audit=read(run/'DATA_AUDIT.json')
        lines=['# 执行状态修正：CPU 实现回交','','READY_FOR_GPU_STAGE_NOT_LAUNCHED','',
               '模型接口、真实标签重算、前向/反向、参数更新、优化器/RNG恢复已运行；方法收益尚未测试。',
               '',f"CPU 契约 {tests['tests']} 项通过；三臂各3次真实FIT批次更新（含普通动作CE）。另有隔离单元测试12次更新，均不用于方法效应或正式初始化。",'',
               f"数据：{audit['unique_source_traces']} 条去重来源轨迹，75派生族、40父族、5屋；6886个缓存特征。FIT4屋32父族，DEV1屋8父族；没有新增独立测试或仿真。",'',
               '|模式|CPU更新|首时刻到末端动作梯度范数|序列长度|', '|---|---:|---:|---:|']
        for row in evidence['real_cached_feature_gradients']:
            lines.append(f"|{row['mode']}|3|{row['early_action_gradient_norm']:.7g}|{row['sequence_length']}|")
        lines+=['','上述非零梯度与状态动作项只证明实现连通，不证明策略在闭环中有益地使用状态。合成撤销用例只检验公式，不是已学会纠错。',
                '', '零GPU、零新底模前向、零新导航。完整GPU训练、现场数值核验、真实自主续接、独立屋与普通VLN验证均未完成。',
                '', '下一关先测DIRECT这一强简单替代与REVISE，MONOTONIC定位累积错误；同轨迹、同初始参数、同更新和普通动作池。若新臂仅提高状态分数而无闭环增量，不保留方法优势主张。']
        (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
        files={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')}
        files.update({str(p.relative_to(HERE)):sha(p) for p in [HERE/'PROTOCOL.json',run/'BINDING.json',run/'DATA_AUDIT.json',run/'CPU_TEST_RESULT.json',run/'RESULT.json',run/'REPORT_ZH.md']})
        write(run/'EVIDENCE_MANIFEST.json',dict(files=files,scope='CPU implementation evidence, not efficacy'),True)
        write(run/'STATUS.json',dict(status=result['status'],gpu_started=False))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write(run/'STATUS.json',dict(status='CPU_PREPARATION_FAILED',error=repr(exc),gpu_started=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=HERE/'runs/cpu_001')
    main(parser.parse_args().run.resolve())
