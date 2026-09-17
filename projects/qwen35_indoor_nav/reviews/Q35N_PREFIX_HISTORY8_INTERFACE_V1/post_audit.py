"""Independent read-only closure of the actual history8 interface diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def main():
    lock=read(HERE/'SOURCE_LOCK.json')
    for path,digest in lock['files'].items():
        resolved=Path(path).resolve(strict=True)
        assert resolved.is_relative_to(LINE.parents[1]) and sha(resolved)==digest,path
    result=read(HERE/'RESULT.json');launch=read(HERE/'LAUNCH_RESULT.json');process=read(HERE/'PROCESS.json')
    assert result['status']=='PASS_INTERFACE_ONLY' and result['old_entry_max_abs']==0.
    assert result['old_entry_argmax_same'] and result['parameters_exact_unchanged']
    assert result['parameter_updates']==result['simulator_actions']==0
    assert result['forward_decisions']==144 and result['backward_calls']==8
    assert launch['status']=='COMPLETE' and launch['error'] is None
    assert launch['cleanup']['exit_code']==0 and launch['cleanup']['exited']
    assert not launch['cleanup']['signals'] and launch['cleanup']['cleanup_error'] is None
    assert not launch['holders_released'] and not launch['other_processes_signaled']
    assert launch['wall_seconds']<=600 and not (Path('/proc')/str(process['pid'])).exists()
    for mode,metrics in result['modes'].items():
        assert metrics['microbatch']==4 and metrics['peak_allocated_bytes']<25*1024**3
        assert len(metrics['single_inference_seconds'])==24
        assert len(metrics['warm_forward_backward_seconds'])==3
        assert len(metrics['gradient_norms'])==4 and all(len(row)==28 for row in metrics['gradient_norms'])
        assert all(math.isfinite(x) for row in metrics['gradient_norms'] for x in row.values())
        actual=read(HERE/'run_001'/(mode+'_LOGITS.json'))
        assert len(actual['samples'])==len(actual['logits'])==24
        assert all(len(row)==4 and all(math.isfinite(v) for v in row) for row in actual['logits'])
    xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    gpu=next(x for x in xml.findall('gpu') if x.findtext('uuid')=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8')
    assert not gpu.findall('processes/process_info'),'GPU1_CURRENTLY_OCCUPIED'
    prefix=read(LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/LAUNCH_FAILURE.json')
    assert 'OWNED_IDENTITY_CHANGED' in prefix['error']
    assert read(LINE/'sft_acceptance/ordinary_stop_row_v12/FEATURE_PARITY.json')['status']=='FAIL'
    report=dict(status='PASS_CLOSED_INTERFACE_ONLY',unix=time.time(),source_files_verified=len(lock['files']),
        source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),result_sha256=sha(HERE/'RESULT.json'),
        launch_sha256=sha(HERE/'LAUNCH_RESULT.json'),cpu_tests_sha256=sha(LINE/'sft_acceptance/ordinary_prefix_history8_v1/CPU_TEST_RESULT.json'),
        original_32_max_abs=0.,new_history_frames=8,trainable_count=28,
        forward_decisions=144,backward_calls=8,parameter_updates=0,simulator_actions=0,
        worker_gone=True,gpu1_currently_empty=True,navigation_gain=False,
        old_failures_unchanged=True,
        speed_ratio_two_over_eight=result['modes']['two_rgb']['warm_training_decisions_per_second']/result['modes']['prefix_eight_rgb']['warm_training_decisions_per_second'],
        compute_only_estimated_three_gpu_hours_384k={k:384000/(3*v['warm_training_decisions_per_second'])/3600 for k,v in result['modes'].items()},
        estimate_excludes_io_communication_optimizer=True)
    with (HERE/'POSTAUDIT.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()

