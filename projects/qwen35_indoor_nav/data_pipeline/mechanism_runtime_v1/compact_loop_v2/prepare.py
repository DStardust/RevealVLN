import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'feedback_generation_v1'
ROOT = HERE.parents[4]

def sha(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT), path
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20), b''): h.update(block)
    return h.hexdigest()

def main():
    old_lock = json.loads((OLD/'run_v1/INPUT_LOCK.json').read_text())
    for path, expected in old_lock.items(): assert sha(path) == expected, path
    old_result = json.loads((OLD/'run_v1/SUPERVISOR_RESULT.json').read_text())
    assert old_result['returncode'] == 0 and old_result['cleanup_complete']
    cfg = json.loads((OLD/'run_v1/EXECUTION_CONFIG.json').read_text())
    cfg['node'] = 'Q35N_COMPACT_LOOP_MECHANISM_GENERATION_V2'
    cfg['revision'] = 'actual_compact_loop_geodesic_start_and_early_neutrality_v2'
    out = HERE/'run_v1'; out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as stream: json.dump(cfg, stream, indent=2)
    paths = [HERE/name for name in ('compact.py','worker.py','run.py','prepare.py','test_compact.py','SPEC_ZH.md')]
    paths += [OLD/'run_v1/INPUT_LOCK.json',OLD/'run_v1/result.json',out/'EXECUTION_CONFIG.json']
    lock = dict(old_lock)
    lock.update({str(path):sha(path) for path in paths})
    with (out/'INPUT_LOCK.json').open('x') as stream: json.dump(lock, stream, indent=2)
    print(json.dumps({'prepared':True,'candidates':len(cfg['candidates']), 'gpu':cfg['gpu_device'], 'input_lock_sha256':sha(out/'INPUT_LOCK.json')}))

if __name__ == '__main__': main()
