"""Freeze the correctness revision before evaluating any nonzero FIT bias."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
HPY=LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'FIT_RESULT.json').exists() and not (HERE/'CALIBRATION.json').exists()
    assert not (HERE/'FIT_METRIC_REVISION_SEAL.json').exists()
    result=subprocess.run([str(HPY),'-I','-B',str(HERE/'test_metric_v2.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    with (HERE/'METRIC_V2_TEST_RESULT.json').open('x') as f:
        json.dump(dict(passed=True,stdout=result.stdout,stderr=result.stderr,unix=time.time(),numerical_tolerance_unchanged=1e-9),f,ensure_ascii=False,indent=2)
    paths=[HERE/name for name in ('counterfactual.py','counterfactual_v2.py','fit.py','fit_v2.py','fit_v3.py','freeze_metric_v2.py','test_metric_v2.py','METRIC_V2_TEST_RESULT.json','FIT_V2_FAILURE.json','METRIC_REVISION_V2.md','SPEC_ZH.md')]
    paths.extend([LINE/'closed_loop_bench/r2r_ce_tiny_v1/metrics.py',ROOT/'third_party/habitat-lab/habitat/tasks/nav/nav.py'])
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (HERE/'FIT_METRIC_REVISION_SEAL.json').open('x') as f:
        json.dump(dict(files=files,unix=time.time(),fit_nonzero_bias_candidates_evaluated_before_freeze=0,dev_data_used=False),f,ensure_ascii=False,indent=2)
    print('FROZEN_METRIC_CORRECTION_WITHOUT_THRESHOLD_CHANGE')


if __name__=='__main__':main()
