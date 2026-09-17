import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
AUTH=ROOT/'projects/qwen35_indoor_nav/authorizations/WITNESS_MULTI_PROGRAM_COHORT_GPU1_V1.json'
spec=importlib.util.spec_from_file_location('batch_03_readiness',HERE.parent/'readiness_v1.py')
ready=importlib.util.module_from_spec(spec);spec.loader.exec_module(ready)
if __name__=='__main__':
    cfg=json.loads((HERE/'run_v1/EXECUTION_CONFIG.json').read_text())
    auth=json.loads(AUTH.read_text())
    assert auth['approved'] and cfg['gpu_device']==1 and not auth['training_allowed']
    assert cfg['source_selection_indices']==auth['batch_indices'][HERE.name]
    lock=json.loads((HERE/'run_v1/INPUT_LOCK.json').read_text())
    for path,h in lock.items():
        digest=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
        assert digest.hexdigest()==h,path
    ready.run_main(HERE)
