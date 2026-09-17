"""Read back real resumed progress, checkpoints, monitors, and frozen source bindings."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ordinary_baseline_v3'
RUN = HERE / 'formal/attempt_001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    protocol = json.loads((HERE / 'PROTOCOL_FILESTORE.json').read_text())
    for name, digest in protocol['code_sha256'].items():
        assert sha(HERE / name) == digest, 'RECOVERY_SOURCE_CHANGED:' + name
    original = json.loads((OLD / 'PROTOCOL_3GPU_SEG2.json').read_text())
    for name, digest in original['code_sha256'].items():
        assert sha(OLD / name) == digest, 'FROZEN_OLD_SOURCE_CHANGED:' + name
    expected_model = (OLD / 'model.py').read_text().replace(
        '        self.exec_embed = nn.Embedding',
        "        rank_device = torch.device('cuda', torch.cuda.current_device())\n        self.exec_embed = nn.Embedding")
    expected_model = expected_model.replace("device='cuda:0'", 'device=rank_device').replace(
        ".to('cuda:0')", ".to(torch.device('cuda', torch.cuda.current_device()))")
    assert expected_model == (HERE / 'model.py').read_text(), 'MODEL_ALGORITHM_NOT_DEVICE_ONLY'
    checks = []
    for step in (30600, 30800):
        path = RUN / ('checkpoint_%09d.pt' % step)
        receipt = json.loads(Path(str(path) + '.json').read_text())
        assert sha(path) == receipt['sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['cursor'] == receipt['cursor'] and state['cursor']['updates'] == step
        assert state['binding']['protocol_sha256'] == sha(HERE / 'PROTOCOL_FILESTORE.json')
        assert state['binding']['sample_index_sha256'] == original['sample_index_sha256']
        assert state['charged_compute_decisions'] == receipt['charged_compute_decisions']
        assert all(torch.isfinite(t).all() for t in state['trainable'].values())
        assert all(torch.isfinite(t).all() for item in state['optimizer']['state'].values()
                   for t in item.values() if isinstance(t, torch.Tensor))
        checks.append(dict(path=str(path), sha256=receipt['sha256'], cursor=receipt['cursor'],
                           charged_compute=receipt['charged_compute_decisions'], tensors_finite=True))
    tests = []
    for name in ('test_control.py', 'test_v3.py'):
        result = subprocess.run([sys.executable, '-I', '-B', str(HERE / name)],
                                capture_output=True, text=True, timeout=60)
        tests.append(dict(script=name, exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0
    progress = json.loads((RUN / 'PROGRESS.json').read_text())
    assert progress['cursor']['updates'] >= 30800 and time.time() - progress['unix'] < 60
    records = [json.loads(s) for s in (RUN / 'PROGRESS.jsonl').read_text().splitlines()]
    assert len(records) >= 20
    assert all(a['cursor']['updates'] < b['cursor']['updates'] for a, b in zip(records, records[1:]))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monitor = json.load(opener.open('http://127.0.0.1:18767/api/status', timeout=5))
    assert monitor['state'] == 'TRAINING' and monitor['read_only']
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader'], text=True)
    root = HERE.parents[3]
    result = dict(status='PASS_RUNTIME_RECOVERY', unix=time.time(),
        git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        code_binding='Explicit per-file SHA seals; repository commit alone does not describe ignored route files',
        protocol_sha256=sha(HERE / 'PROTOCOL_FILESTORE.json'),
        old_checkpoint_preserved=True, old_uncheckpointed_reported_updates_replayed=99,
        additional_unlogged_legacy_compute='UNKNOWN; conservative reserve kept in budget, not measured usage',
        legacy_cpu_transport_failures=[
            dict(kind='torchrun standalone TCPStore hostname timeout', exit=1, gpu_updates=0),
            dict(kind='static loopback TCPStore timeout', exit=1, gpu_updates=0),
            dict(kind='static loopback USE_LIBUV=0 TCPStore timeout', exit=1, gpu_updates=0),
            dict(kind='single-process TCPStore(use_libuv=False) timeout', exit=1, gpu_updates=0)],
        real_distributed_test=json.loads((HERE / 'cpu_filestore_v1/RESULT.json').read_text()),
        cpu_tests=tests, checkpoint_checks=checks, progress=progress,
        monitor_state=monitor['state'], monitor_address='http://127.0.0.1:18767',
        gpu_snapshot=gpu, scientific_pass=False, navigation_gain_verified=False,
        model_architecture_changed=False, model_device_binding_corrected=True,
        original_input_and_source_seals_verified=True,
        next_action='Continue existing bounded supervisor; do not launch a duplicate')
    with (HERE / 'LIVE_ACCEPTANCE.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], updates=progress['cursor']['updates'],
                         checkpoints_verified=[30600, 30800], monitor=monitor['state'])), flush=True)


if __name__ == '__main__':
    main()
