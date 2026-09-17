"""Explicit new-level GPU1 supervisor adapter; draft cannot launch without approval."""
import types
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import RUNTIME,LINE,runtime_config,save

def adapted_source():
    source=(RUNTIME/'compact_loop_v2/run.py').read_text()
    old="assert sample['elapsed'] < 3000, 'WALL_BUDGET'"
    new="assert sample['elapsed'] < 4500, 'WALL_BUDGET'"
    assert source.count(old)==1
    source=source.replace(old,new)
    old="sample['disk_bytes'] < 7*1024**3"
    assert source.count(old)==1
    source=source.replace(old,"sample['disk_bytes'] < 8*1024**3")
    # Persist even the snapshot that triggers a stop; preserve all observed external contexts.
    old="snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)"
    assert source.count(old)==1
    source=source.replace(old,"snapshot=gpu()\n                with (OUT/'GPU_SNAPSHOTS.jsonl').open('a') as stream: stream.write(json.dumps(snapshot)+'\\n')\n                upper=check_gpu(snapshot,proc.pid)")
    return source

def main():
    cfg=runtime_config()
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    save(out/'EXECUTION_CONFIG.json',cfg)
    module=types.ModuleType('scout_gpu1_supervisor')
    module.__file__=str(RUNTIME/'compact_loop_v2/run.py')
    exec(compile(adapted_source(),module.__file__,'exec'),module.__dict__)
    module.HERE=HERE;module.OUT=out;module.LINE=LINE
    module.ENV=LINE/'.envs/q35n_habitat_v017_g0r'
    assert module.UUID=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'
    module.main()

if __name__=='__main__':main()
