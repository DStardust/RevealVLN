"""Monitor only this node's explicitly admitted child, without stopping holders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent.parent
LINE = HERE.parents[1]
ENV = LINE/'.envs/q35n_habitat_v017_g0r'


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def gpu_state(index):
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-i', str(index), '-q', '-x'], text=True, timeout=15))
    gpu = root.find('gpu')
    return {'index': index, 'uuid': gpu.findtext('uuid'),
            'memory_mib': float(gpu.findtext('fb_memory_usage/used').split()[0]),
            'utilization': float(gpu.findtext('utilization/gpu_util').split()[0]),
            'processes': [{x.tag: x.text for x in p} for p in gpu.findall('processes/process_info')]}


def process_tree_memory(pid):
    # Process identities only; no task files or environment reads.
    raw = subprocess.check_output(['ps', '-e', '-o', 'pid=,ppid=,rss='], text=True, timeout=10)
    rows = [list(map(int, line.split())) for line in raw.splitlines() if line.strip()]
    wanted = {pid}
    while True:
        added = {p for p, parent, _ in rows if parent in wanted} - wanted
        if not added:
            break
        wanted |= added
    return sum(rss*1024 for p, _, rss in rows if p in wanted), sorted(wanted)


def disk_bytes(root):
    total = 0
    for p in root.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


def owned_session_members(session_id):
    raw = subprocess.check_output(['ps', '-e', '-o', 'pid=,pgid=,sid=,stat='], text=True, timeout=10)
    rows = [line.split() for line in raw.splitlines() if line.strip()]
    return [(int(pid), int(group)) for pid, group, sid, state in rows
            if int(sid) == session_id and not state.startswith('Z')]


def stop_owned(proc):
    # Reap the registered parent, but do not assume its exit killed its children.
    members = owned_session_members(proc.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted({g for _, g in members}):
            current = owned_session_members(proc.pid)
            if any(g == group for _, g in current):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+5
        while members and time.monotonic() < deadline:
            proc.poll()
            time.sleep(.1)
            members = owned_session_members(proc.pid)
        if not members:
            break
    proc.wait(timeout=10)
    return members


def run(out):
    from guard import ResourceGuard, BudgetError
    cfg = json.loads((out/'EXECUTION_CONFIG.json').read_text())
    auth = json.loads((out/'EXECUTION_AUTH.json').read_text())
    verify_admission(out, cfg, auth)
    before = gpu_state(cfg['gpu_device'])
    assert before['index'] == 2 and before['uuid'] == 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
    assert before['memory_mib'] < 1024 and before['utilization'] == 0, 'DEVICE_NOT_IDLE'
    existing = {int(p['pid']) for p in before['processes']}
    assert existing <= {2781015, 2900192}, 'UNREVIEWED_GPU_PROCESS'
    for pid in existing:
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'args='], text=True, timeout=10)
        assert 'eval.scripts.evaluate_pointgoal' in command and '--device cuda:0' in command
    save(out/'GPU_LEASE_BEFORE.json', {**before, 'no_process_stopped': True,
         'basis': 'ancillary contexts of verified cuda:0 jobs, GPU2 utilization zero'})
    cache = out/'cache'
    cache.mkdir()
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CONDA_PREFIX', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(cache), TMPDIR=str(cache), NUMBA_CACHE_DIR=str(cache/'numba'),
               MPLCONFIGDIR=str(cache/'mpl'), CUDA_CACHE_PATH=str(cache/'cuda'),
               __GL_SHADER_DISK_CACHE_PATH=str(cache/'gl'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Reserve 2 GiB below the absolute disk ceiling for in-flight writes/cleanup.
    guard = ResourceGuard(cfg['budget']['total_seconds'], cfg['ram_cap_bytes'], 14*1024**3)
    started, samples, error, proc = time.monotonic(), [], None, None
    known_owned, cleanup_members = set(), []
    cleanup_errors, after = [], None
    def sample():
        g = gpu_state(cfg['gpu_device'])
        pids = {os.getpid()} | {pid for pid, _ in owned_session_members(proc.pid)}
        known_owned.update(pids-{os.getpid()})
        raw = subprocess.check_output(['ps', '-e', '-o', 'pid=,rss='], text=True, timeout=10)
        rss = sum(int(line.split()[1])*1024 for line in raw.splitlines() if int(line.split()[0]) in pids)
        s = {'wall_seconds': time.monotonic()-started, 'ram_bytes': rss,
             'disk_bytes': disk_bytes(out), 'gpu': g, 'own_tree_pids': sorted(pids)}
        samples.append(s)
        guard.check(s)
        if g['memory_mib'] >= cfg['gpu_cap_mib']:
            raise BudgetError('gpu_mib', g['memory_mib'], cfg['gpu_cap_mib'])
        assert {int(p['pid']) for p in g['processes']} <= existing | known_owned, 'NEW_EXTERNAL_GPU_WORK'
    with (out/'worker.log').open('x') as log:
        try:
            proc = subprocess.Popen([str(ENV/'bin/python3'), '-I', '-B', str(HERE/'p0_driver_v1/worker.py'), '--output', str(out)],
                                    cwd=LINE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(out/'OWN_PROCESS.json', {'pid': proc.pid, 'pgid': proc.pid, 'command_scope': 'this_node_worker_only'})
            known_owned.add(proc.pid)
            while proc.poll() is None:
                sample()
                time.sleep(1)
            sample()
        except BaseException as exc:
            error = {'message': repr(exc), 'resource_censored': isinstance(exc, BudgetError)}
        finally:
            try:
                if proc is not None:
                    known_owned.update(pid for pid, _ in owned_session_members(proc.pid))
                    cleanup_members = stop_owned(proc)
            except BaseException as exc:
                cleanup_errors.append({'stage': 'own_session_cleanup', 'error': repr(exc)})
            try:
                after = gpu_state(cfg['gpu_device'])
            except BaseException as exc:
                cleanup_errors.append({'stage': 'final_gpu_query', 'error': repr(exc)})
            remaining = (bool(cleanup_errors) or after is None or bool(cleanup_members)
                         or any(int(p['pid']) in known_owned for p in after['processes']))
            save(out/'GPU_LEASE_AFTER.json', after or {'state': 'UNKNOWN'})
            save(out/'RESOURCE_SAMPLES.json', samples)
            save(out/'GPU_RESTORE_RESULT.json', {'stopped_placeholders': [], 'restoration_required': False,
                'own_process_remaining': remaining, 'cleanup_complete': not remaining,
                'no_external_process_signals_sent': True, 'cleanup_errors': cleanup_errors})
            final_disk = disk_bytes(out)
            if final_disk > cfg['disk_cap_bytes']:
                error = {'message': 'FINAL_DISK_CAP', 'resource_censored': True}
            save(out/'SUPERVISOR_RESULT.json', {'returncode': proc.returncode if proc else None,
                 'error': error, 'wall_seconds': time.monotonic()-started, 'cleanup_complete': not remaining,
                 'new_placeholder_created': False, 'sampled_limits_only': True,
                 'final_disk_bytes_before_result': final_disk, 'cleanup_errors': cleanup_errors})
    return 0 if proc is not None and proc.returncode == 0 and error is None and not remaining else 1


def verify_admission(out, cfg, auth):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert auth['approved'] and auth['P0_authorized'] and cfg['runtime_allowed']
    assert cfg['phase'] == 'five_fixed_bundle_constructability_P0'
    assert sha(out/'EXECUTION_CONFIG.json') == auth['config_sha256']
    expected = {str(p.relative_to(HERE)) for p in HERE.glob('*.py')}
    expected |= {str(p.relative_to(HERE)) for p in (HERE/'p0_driver_v1').glob('*.py')}
    assert set(auth['code_sha256']) == expected, 'INCOMPLETE_CODE_LOCK'
    for name, value in auth['code_sha256'].items():
        assert sha(HERE/name) == value, 'CODE_CHANGED:'+name
    assert sha(HERE/'smoke_v1/result.json') == cfg['smoke_result_sha256']
    assert json.loads((HERE/'smoke_v1/result.json').read_text())['runtime_pass']
    supervisor = json.loads((HERE/'smoke_v1/SUPERVISOR_RESULT.json').read_text())
    assert supervisor['returncode'] == 0 and supervisor['error'] is None and supervisor['cleanup_complete']
    assert cfg['budget'] == {'total_actions': 200000, 'total_seconds': 6000,
                            'discovery_actions': 20000, 'discovery_seconds': 720,
                            'certification_actions': 20000, 'certification_seconds': 480}
    assert cfg['disk_cap_bytes'] == cfg['ram_cap_bytes'] == 16*1024**3 and cfg['gpu_cap_mib'] == 8192
    assert [r['candidate_id'] for r in cfg['candidates']] == ['MP5_%02d' % i for i in range(5)]
    binary = (ENV/'bin/python3').resolve()
    assert binary.is_relative_to(LINE) and binary == Path(auth['python_realpath'])
    assert sha(binary) == auth['python_sha256']


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(HERE) and out != HERE
    raise SystemExit(run(out))
