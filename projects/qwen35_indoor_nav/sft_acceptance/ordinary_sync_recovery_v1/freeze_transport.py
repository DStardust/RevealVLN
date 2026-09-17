"""Versioned transport amendment after CPU TCPStore failures; no old seals rewritten."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2)


def main():
    tests = json.loads((HERE / 'cpu_filestore_v1/RESULT.json').read_text())
    assert tests['status'] == 'PASS'
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    for name, digest in protocol['code_sha256'].items():
        assert sha(HERE / name) == digest, name
    new_names = ['train_filestore.py', 'supervise_filestore.py', 'launcher.py', 'test_file_store.py',
                 'freeze_transport.py']
    protocol['code_sha256'].update({n: sha(HERE / n) for n in new_names})
    protocol['transport_amendment'] = dict(store='fresh FileStore per attempt',
        process_group='NCCL 180 seconds', interfaces='lo for single-node bootstrap',
        reason='TCPStore client validation timed out on both hostname and 127.0.0.1, libuv variants; raw loopback TCP and three-rank FileStore Gloo passed.',
        old_seal_preserved=sha(HERE / 'PROTOCOL.json'), no_gpu_attempt_before_amendment=True)
    save('PROTOCOL_FILESTORE.json', protocol)
    runbook = json.loads((HERE / 'RUNBOOK.json').read_text())
    runbook['code_sha256'] = protocol['code_sha256']
    runbook['steps'][0]['argv'][-1] = str(HERE / 'supervise_filestore.py')
    runbook['lease_wall_seconds'] = int(protocol['accounting']['deadline_unix'] - time.time()) + 120
    save('RUNBOOK_FILESTORE.json', runbook)
    save('MAIN_AGENT_APPROVAL.json', dict(unix=time.time(),
        scope='Ordinary baseline reliability repair and continuation only; no architecture novelty experiment',
        protocol_sha256=sha(HERE / 'PROTOCOL_FILESTORE.json'),
        runbook_sha256=sha(HERE / 'RUNBOOK_FILESTORE.json'),
        cpu_unit_tests_passed=16, actual_three_rank_test=tests,
        old_failure_preserved=True, maximum_attempts=3, gpu_indices=[3, 4, 5],
        training_resume_allowed=True, scientific_gain_verified=False))
    print('FILESTORE_AMENDMENT_FROZEN_AND_APPROVED', flush=True)


if __name__ == '__main__':
    main()
