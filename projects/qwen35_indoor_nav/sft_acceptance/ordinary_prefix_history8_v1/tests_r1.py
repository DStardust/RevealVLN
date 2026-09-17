"""CPU-only real input, deployed Window, processor, sampling and child-lifetime checks."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
STDPY = ROOT / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
h = runpy.run_path(str(HERE / 'history.py'))
o = runpy.run_path(str(HERE / 'owned_process_r1.py'))
d = runpy.run_path(str(HERE / 'data.py'))

def process_tests():
    checks = []
    def child(code):
        argv = [str(STDPY), '-I', '-S', '-B', '-c', code]
        p = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        return p, o['OwnedChild'](p, argv, ROOT)
    # Real live child and normal completion, including zombie visibility before poll.
    p, owned = child('import sys; sys.stdin.buffer.read(1)')
    assert owned.check() == 'live'
    p.stdin.write(b'x'); p.stdin.flush()
    for _ in range(100):
        row = o['inspect'](p.pid)
        if row and row['state'] == 'Z':
            break
        time.sleep(.005)
    assert owned.check() == 'exited'
    receipt = owned.cleanup()
    assert receipt['exit_code'] == 0 and receipt['signals'] == [] and receipt['cleanup_error'] is None
    checks.append('normal_exit_and_zombie_reap')

    # Force natural exit between the initial poll and identity read.
    p, owned = child('import sys; sys.stdin.buffer.read(1)')
    def exit_during_read(pid):
        p.stdin.write(b'x'); p.stdin.flush(); p.wait(timeout=2)
        return None
    assert owned.check(reader=exit_during_read) == 'exited'
    assert owned.cleanup()['signals'] == []
    checks.append('poll_procfs_exit_race')

    # Missing procfs alone is not evidence of completion.
    p, owned = child('import sys; sys.stdin.buffer.read(1)')
    assert owned.check(reader=lambda _: None) == 'exiting'
    assert p.poll() is None
    receipt = owned.cleanup()
    assert receipt['exited'] and receipt['signals'] == [signal.SIGTERM]
    checks.append('transient_missing_is_not_terminal')

    # Reused start time and a live exec identity change both refuse signals.
    for field in ('start', 'argv', 'cwd'):
        p, owned = child('import sys; sys.stdin.buffer.read(1)')
        row = o['inspect'](p.pid)
        row[field] = row[field] + 1 if field == 'start' else (['unknown-executable'] if field == 'argv' else '/unknown')
        try:
            owned.check(reader=lambda _: row)
            raise AssertionError('IDENTITY_MISMATCH_ACCEPTED')
        except o['IdentityConflict']:
            pass
        assert owned.signals == [] and p.poll() is None
        receipt = owned.cleanup()
        assert receipt['exited'] and receipt['cleanup_error'] is None
        checks.append('refuse_live_' + field)

    # A child ignoring TERM gets a bounded KILL, still via its pidfd.
    argv = [str(STDPY), '-I', '-S', '-B', '-c',
            'import signal,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); sys.stdin.buffer.read(1)']
    p = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert p.stdout.readline() == b'ready\n'
    owned = o['OwnedChild'](p, argv, ROOT)
    receipt = owned.cleanup(timeout=.05)
    assert receipt['exited'] and receipt['signals'] == [signal.SIGTERM, signal.SIGKILL] and receipt['cleanup_error'] is None
    checks.append('bounded_term_then_kill')
    return checks

def main():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_REQUIRED'
    assert not torch.cuda.is_initialized()
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    for t in range(10000):
        length = max(8, t + 1); pad = length - t - 1
        reference = [None if int(x) - pad < 0 else int(x) - pad for x in np.linspace(0, length - 1, 7, endpoint=False, dtype=int)] + [t]
        assert h['indices'](t) == reference
        assert all(i is None or 0 <= i <= t for i in reference)
        # Future observations, even if present in a source file, cannot be selected.
        prefix = list(range(t + 1))
        assert h['choose'](prefix, t, None) == h['choose'](prefix + ['FUTURE'], t, None)
    assert h['indices'](0) == [None] * 7 + [0]
    for t in (-1, 1.0, True):
        try:
            h['indices'](t); raise AssertionError('INVALID_T_ACCEPTED')
        except ValueError:
            pass
    process_checks = process_tests()
    rows, report = d['load_rows']()
    assert len(rows) == 37114 and report['decisions_per_epoch'] == 2650347
    groups = {}
    for idx, row in enumerate(rows):
        if 16 <= row['decisions'] <= 200 and row['source'] not in groups:
            groups[row['source']] = idx
    assert len(groups) == 3, groups
    # Also test longest route's offline prefix; deployment still has its 500 budget.
    selected = list(groups.values()) + [max(range(len(rows)), key=lambda i: rows[i]['decisions'])]
    store = d['SampleStore'](rows)
    model = runpy.run_path(str(LINE / 'sft_acceptance/ordinary_sync_recovery_v1/model.py'))
    processor = AutoProcessor.from_pretrained(model['MODEL'], local_files_only=True, trust_remote_code=False)
    ids = json.loads((LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens': list(ids)})
    assert all(processor.tokenizer.convert_tokens_to_ids(k) == v for k, v in ids.items())
    # Actual configured image ID, never trust a hand-coded ID.
    config = json.loads((model['MODEL'] / 'config.json').read_text())
    collate = model['make_collate'](processor.tokenizer.pad_token_id,
        {a: ids[model['EXEC_TOKENS'][a]] for a in model['ACTIONS'][:-1]},
        ids['<NAV_ACTION_QUERY>'], config['image_token_id'])
    samples = []
    records = []
    tensor_checks = 0
    for idx in selected:
        record = store.record(idx)
        times = sorted({0, 1, 7, 8, len(record)//2, len(record)-1})
        for t in times:
            item = store.get(idx, t)
            assert set(item) == {'instruction', 'images', 'executed', 'target'}
            assert item['target'] == model['ACTIONS'].index(record.actions[t])
            assert item['executed'] == list(record.actions[max(0,t-8):t])
            assert len(item['images']) == 8
            assert item['images'][-1].tobytes() == d['decoder']['load_rgb'](d['decoder']['_relative'](record.rgb_root, record._refs[t])).tobytes()
            samples.append(dict(record_idx=idx,t=t,target=item['target'],weight=1.))
        if idx not in groups.values():
            continue
        window = h['PrefixWindow']()
        for t in range(len(record)):
            image = d['decoder']['load_rgb'](d['decoder']['_relative'](record.rgb_root, record._refs[t]))
            payload = dict(done=False, rgb=base64.b64encode(image.tobytes()).decode())
            if t == 0:
                payload['instruction'] = record.instruction
            assert window.receive(payload, executed=None if t == 0 else record.actions[t-1])
            if t not in times:
                continue
            offline = store.get(idx,t)
            online = window.item()
            assert set(online) == {'instruction','images','executed'}
            assert online['instruction'] == offline['instruction']
            assert online['executed'] == offline['executed']
            assert [x.tobytes() for x in online['images']] == [x.tobytes() for x in offline['images']]
            assert window.input_audit()['input_frame_indices'] == h['indices'](t)
            sample = [dict(record_idx=idx,t=t,target=offline['target'],weight=1.)]
            class OnlineStore:
                def get(self, _idx, _t):
                    return online
            a = model['DecisionDataset'](sample,store,processor)
            b = model['DecisionDataset'](sample,OnlineStore(),processor)
            aa = collate([a[0]]); bb = collate([b[0]])
            with torch.inference_mode():
                cc = collate([a[0]])
            assert aa.keys() == bb.keys() == cc.keys()
            assert all(torch.equal(aa[k],bb[k]) and torch.equal(aa[k],cc[k]) for k in aa)
            tensor_checks += 1
        assert not window.receive(dict(done=True), executed='STOP')
        bad = dict(done=False,rgb=base64.b64encode(bytes(h['RGB_BYTES'])).decode(),instruction='new',goal=[0,0,0])
        try:
            window.receive(bad); raise AssertionError('PRIVILEGED_INPUT_ACCEPTED')
        except ValueError:
            pass
        del bad['goal']; assert window.receive(bad)
        assert len(window.frames) == 1 and not window.executed
        assert window.input_audit()['padding_mask'] == [True]*7+[False]
        records.append(dict(record_idx=idx,source=rows[idx]['source'],decisions=len(record)))
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS',unix=time.time(),sampling_comparisons=10000,process_checks=process_checks,
                  snapshot_report=report,actual_records=records,offline_samples=len(samples),
                  offline_online_and_inference_context_tensor_comparisons=tensor_checks,
                  gpu_launches=0,parameter_updates=0,simulator_actions=0,navigation_gain=False)
    with (HERE / 'CPU_TEST_RESULT.json').open('x') as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
    with (HERE / 'DIAGNOSTIC_SAMPLES.json').open('x') as out:
        json.dump(samples,out,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()


