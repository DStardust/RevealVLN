"""Reproduce the late-import failure under a real legacy-objective import."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    cfg=read(HERE/'PROTOCOL.json');raw=read(Path(cfg['eval_run'])/'DATA.json')['raw_families']
    # Match the prepare.py load sequence; this prepends V16 to sys.path.
    common=load('import_regression_base_common',PARENT/'common.py');sys.modules['common']=common
    load('import_regression_objective',PARENT/'objective.py')
    sys.modules.pop('evaluate_continuations',None)
    import evaluate_continuations as wrong
    assert Path(wrong.__file__).parent==V16
    try:wrong.registry_value(raw,cfg)
    except ValueError as exc:assert str(exc)=='EVALUATION_FAMILY_COUNT'
    else:raise AssertionError('ORIGINAL_FAILURE_NOT_REPRODUCED')
    local=local_module('evaluate_continuations');reg=local.registry_value(raw,cfg)
    assert Path(local.__file__).resolve()==HERE/'evaluate_continuations.py'
    assert len(reg['models'])==6 and len(reg['conditions'])==256 and len(reg['slots'])==1536
    assert sum(c['available'] for c in reg['conditions'])==248
    encoder=local_module('encoder');assert Path(encoder.__file__).parent==HERE
    assert encoder.RawStore.__module__=='scale_local_encoder'
    assert local_module('review').registry.__globals__['__file__']==str(HERE/'evaluate_continuations.py')
    write(HERE/'IMPORT_REGRESSION_R3.json',dict(status='PASS',old_failure_reproduced=True,local_module=str(local.__file__),models=6,planned_slots=1536,available_slots=1488,numerical_or_metric_changes=False,gpu_loaded=False))
    print('Late import failure reproduced; explicit local registry/encoder/review all passed.',flush=True)
if __name__=='__main__':main()
