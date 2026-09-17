"""Targeted regression of the live-directory scan fix and current lease entry."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
def main():
    runtime=runpy.run_path(str(HERE/'runtime.py'))
    scan=runtime['load']('live_scan_regression',LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/tree_size.py')
    temp=Path(tempfile.mkdtemp(prefix='scan_',dir=LINE/'.th8p'))
    child=temp/'vanishing';child.mkdir()
    original=scan.os.scandir
    def racing(path):
        if Path(path)==child:
            # Only remove this exact, empty, test-created directory.
            child.rmdir()
            raise FileNotFoundError(child)
        return original(path)
    scan.os.scandir=racing
    try:size,misses=scan.tree_size(temp)
    finally:scan.os.scandir=original
    assert size==0 and misses==1
    temp.rmdir()
    try:scan.tree_size(temp);raise AssertionError('MISSING_ROOT_ACCEPTED')
    except FileNotFoundError:pass
    for name in ('supervise.py','lease.py','accept.py','train.py'):
        module=runpy.run_path(str(HERE/name),run_name='CPU_IMPORT_ONLY')
        assert callable(module['main']) and module['HERE']==HERE
    supervisor=runpy.run_path(str(HERE/'supervise.py'),run_name='CPU_IMPORT_ONLY')
    assert supervisor['scan'].__file__==str(LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/tree_size.py')
    assert os.getpid() in supervisor['tree'](os.getpid())
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    receipt=json.loads((HERE/'CPU_TRAINING_ENTRY_RESULT.json').read_text())
    assert receipt['status']=='PASS_CPU_TRAINING_ENTRY'
    result=dict(status='PASS_TRANSPORT_R1_REGRESSION',unix=time.time(),concurrent_directory_enoent_counted=True,
        missing_root_rejected=True,actual_imports=['train.py','supervise.py','lease.py','accept.py'],
        cpu_loader_parent_receipt_sha256=runtime['sha'](HERE/'CPU_TRAINING_ENTRY_RESULT.json'),
        prior_cpu_loader_nfs_finalizer_warnings=True,prior_cpu_loader_workers_joined=True,
        preserved_prior_gpu_transport_failure=True,gpu_launches=0,training_updates=0,
        current_sources={str(p):runtime['sha'](p) for p in HERE.glob('*.py')})
    with (HERE/'TRANSPORT_R1_REGRESSION.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='current_sources'}),flush=True)
if __name__=='__main__':main()

