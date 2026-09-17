"""Independent multi-house batch adapter. No shared frozen-source writes."""
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
GPUS={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def worker_source(source):
    adapter=load('batch_v2_export_adapter',WF/'short_revisit_v2/worker.py')
    source=adapter.source_adapter(source)
    old="components=row['components'])"
    assert source.count(old)==1
    source=source.replace(old,"components=row['components'],balance=row['balance'])")
    source=source.replace("'short_continuation_completed_subgoal_revisit_v2'","'witness_bank_balanced_batch_v1'")
    return source

def worker_main(batch):
    batch=Path(batch).resolve();assert batch.parent==HERE.resolve()
    cfg=json.loads((batch/'run_v1/EXECUTION_CONFIG.json').read_text())
    path=WF/'assembly_v1/worker.py'
    module=types.ModuleType('witness_bank_batch_worker');module.__file__=str(path)
    # Load original method import through its registered adapter before exec.
    load('batch_import_prerequisites',WF/'short_revisit_v2/method.py')
    exec(compile(worker_source(path.read_text()),str(HERE/'shared.py'),'exec'),module.__dict__)
    variants={'lr_v2':WF/'batch_plan_v1/balanced_v2.py','winding_v1':WF/'winding_balance_v1/method.py'}
    method=load('witness_bank_batch_method',variants[cfg['factory_variant']])
    module.HERE=batch;module.WitnessFactory=method.BalancedFactory
    module.main()

def supervisor_source(source,gpu):
    assert gpu in GPUS
    substitutions=[("'nvidia-smi','-i','1'",f"'nvidia-smi','-i','{gpu}'"),
        ("sample['elapsed'] < 3000","sample['elapsed'] < 3900")]
    for old,new in substitutions:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

def run_main(batch):
    batch=Path(batch).resolve();assert batch.parent==HERE.resolve()
    cfg=json.loads((batch/'run_v1/EXECUTION_CONFIG.json').read_text())
    assert cfg['supervision_wall_seconds']==3900 and cfg['budget']['total_seconds']==3600
    gpu=cfg['gpu_device'];assert cfg['gpu_uuid']==GPUS[gpu]
    path=RUNTIME/'compact_loop_v2/run.py'
    module=types.ModuleType('witness_bank_batch_supervisor');module.__file__=str(path)
    exec(compile(supervisor_source(path.read_text(),gpu),str(path),'exec'),module.__dict__)
    module.HERE=batch;module.OUT=batch/'run_v1';module.LINE=RUNTIME.parents[1]
    module.ENV=module.LINE/'.envs/q35n_habitat_v017_g0r';module.UUID=GPUS[gpu]
    module.main()
