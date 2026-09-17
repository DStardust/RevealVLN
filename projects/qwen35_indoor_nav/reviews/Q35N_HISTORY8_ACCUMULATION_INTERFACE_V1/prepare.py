"""Actual CPU imports, source assembly, input provenance and GPU-budget seal."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import time
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
H8=LINE/'sft_acceptance/ordinary_prefix_history8_v1'
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()
def write(path,obj):
    with path.open('x') as stream:json.dump(obj,stream,indent=2)
def main():
    assert not (HERE/'SOURCE_LOCK.json').exists()
    cpu=json.loads((H8/'CPU_TEST_RESULT.json').read_text());assert cpu['status']=='PASS' and len(cpu['process_checks'])==7
    fit=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
    with (HERE/'PROTOCOL.json').open('xb') as stream:stream.write((fit/'PROTOCOL.json').read_bytes())
    (HERE/'run_001').mkdir()
    source=(HERE/'worker.py').read_text()
    source=source[:source.index('exec(compile(source,')]
    namespace={'__file__':str(HERE/'worker.py')}
    exec(compile(source,'CPU_WORKER_ASSEMBLY','exec'),namespace)
    ast.parse(namespace['source'])
    assert 'subprocess.Popen' not in namespace['source'] and 'build_sim' not in namespace['source']
    common=runpy.run_path(str(HERE/'common.py'))
    assert common['TINY']==LINE/'closed_loop_bench/r2r_ce_tiny_v1'
    launcher=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_IMPORT_ONLY')
    assert launcher['LINE']==LINE and callable(launcher['main'])
    for path in list(HERE.glob('*.py'))+list(H8.glob('*.py')):ast.parse(path.read_text())
    files=dict(json.loads((LINE/'reviews/Q35N_PREFIX_HISTORY8_INTERFACE_V1/SOURCE_LOCK.json').read_text())['files'])
    data=runpy.run_path(str(H8/'data.py'))
    rows,report=data['load_rows']()
    samples=json.loads((H8/'DIAGNOSTIC_SAMPLES.json').read_text())
    store=data['SampleStore'](rows)
    pixels=set()
    for sample in samples:
        record=store.record(sample['record_idx'])
        for path in (record.policy_path,record.supervision_path):files[str(path)]=sha(path)
        # Include both original last2 and new prefix8 view pixels.
        idx=set(i for i in data['history']['indices'](sample['t']) if i is not None)
        idx.update(range(max(0,sample['t']-1),sample['t']+1))
        for t in idx:
            pixels.add(data['decoder']['_relative'](record.rgb_root,record._refs[t]))
    for path in pixels:files[str(path)]=sha(path)
    cache=[]
    for path in (fit/'run_001/cache/triton').rglob('*.autotune.json'):
        target=HERE/'triton_cache'/path.relative_to(fit/'run_001/cache/triton')
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(path.read_bytes())
        cache.append(dict(source=str(path),initial_target=str(target),sha256=sha(path)))
        files[str(path)]=sha(path)
    write(HERE/'CACHE_PROVENANCE.json',dict(files=cache,original_cache_unchanged=True))
    write(HERE/'CPU_TEST_RESULT.json',dict(status='PASS',unix=time.time(),actual_prefix_assembled=True,
        actual_common_and_launcher_import=True,source_history_cpu_receipt_sha256=sha(H8/'CPU_TEST_RESULT.json'),
        sampled_ordinary_pixels=len(pixels),simulator_launch_assembled=False,gpu_launches=0))
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'PLAN_ZH.md']+list(H8.glob('*.py'))+list(H8.glob('*.json'))+[H8/'PLAN_ZH.md']+list((H8/'references').iterdir()):
        files[str(path)]=sha(path)
    for path in [LINE/'sft_acceptance/ordinary_stop_row_v12/FEATURES.pt',
                 LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/checks.py',
                 LINE/'sft_acceptance/ordinary_expanded_v1/data.py',
                 LINE/'sft_acceptance/ordinary_expanded_v1/reuse.py']:
        files[str(path)]=sha(path)
    for path in list((LINE/'sft_acceptance/ordinary_history8_paired_train_v1').glob('*.py')) + [LINE/'sft_acceptance/ordinary_onpolicy_fp32_master_v8/initial_fp32_master_from_best4000.pt']:
        files[str(path)]=sha(path)
    write(HERE/'SOURCE_LOCK.json',dict(files=files,unix=time.time(),gpu_seconds=600,parameter_updates=0))
    common['verify_lock']()
    print(json.dumps(dict(status='PASS_FROZEN',files=len(files),ordinary_pixels=len(pixels))),flush=True)
if __name__=='__main__':
    main()


