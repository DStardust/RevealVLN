"""Multi-GPU exact-holder lease runner for v3 acceptance. Stdlib only.

Generalizes the approved single-GPU lease pattern (identity re-read live from
each exact tmux pane, double /proc checks, SIGTERM to exact holder PIDs only,
GPU drain evidence, own-child cleanup, finally respawn every pane with the same
argv/cwd, verify identity + GPU re-occupancy). The runbook is a frozen JSON whose
hash is verified before any signal. No PID from any old document is trusted.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
HOLDER_MIN_MIB = 20000
EXTERNAL_PER_PROCESS_MAX_MIB = 768
EXTERNAL_TOTAL_MAX_MIB = 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def call(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def pane_value(pane, fmt):
    return call('tmux', 'display-message', '-p', '-t', pane, fmt)


def proc(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text()
    rest = stat.rsplit(')', 1)[1].split()
    return dict(pid=pid, state=rest[0], starttime_ticks=int(rest[19]),
                command=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip(),
                cwd=str((root / 'cwd').resolve(strict=True)),
                proc_uid=int(next(x for x in (root / 'status').read_text().splitlines()
                                  if x.startswith('Uid:')).split()[1]))


def proc_status(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw.rsplit(')', 1)[1].split()
    return dict(state=fields[0], starttime_ticks=int(fields[19]))


def gpu_snapshot(uuid):
    out = subprocess.check_output(
        ['nvidia-smi', '-i', uuid, '--query-compute-apps=pid,used_memory',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = {}
    for line in out.strip().splitlines():
        if line.strip():
            pid, mib = [x.strip() for x in line.split(',')]
            processes[int(pid)] = int(mib)
    used = subprocess.check_output(
        ['nvidia-smi', '-i', uuid, '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    return dict(uuid=uuid, processes=processes, memory_mib=int(used))


def holder_identity(holder):
    pane = holder['pane']
    require(pane_value(pane, '#{pane_id}') == holder['pane_id'], 'PANE_IDENTITY_CHANGED')
    require(pane_value(pane, '#{pane_dead}') == '0', 'HOLDER_PANE_DEAD')
    require(pane_value(pane, '#{pane_current_path}') == str(ROOT), 'PANE_CWD')
    pid = int(pane_value(pane, '#{pane_pid}'))
    identity = proc(pid)
    require(identity['command'] == holder['command'], 'HOLDER_COMMAND_MISMATCH')
    require(identity['cwd'] == str(ROOT) and identity['proc_uid'] == 0, 'HOLDER_CWD_UID_MISMATCH')
    identity.update(pane=pane, pane_id=holder['pane_id'], gpu_uuid=holder['gpu_uuid'])
    return identity


def identity_equal(actual, frozen):
    return all(actual.get(k) == frozen.get(k)
               for k in ('pid', 'starttime_ticks', 'command', 'cwd', 'pane', 'pane_id', 'proc_uid'))


def wait_holder_terminal(identity):
    for _ in range(80):
        status = proc_status(identity['pid'])
        if status is None or status['state'] in ('Z', 'X'):
            return
        require(status['starttime_ticks'] == identity['starttime_ticks'], 'HOLDER_PID_REUSED')
        time.sleep(.25)
    raise ValueError('HOLDER_EXIT_FAILED_NO_ESCALATION')


def wait_gpu_drain(out, identity, name):
    for index in range(81):
        status = proc_status(identity['pid'])
        require(status is None or status['state'] in ('Z', 'X'), 'GPU_DRAIN_HOLDER_NOT_TERMINAL')
        snapshot = gpu_snapshot(identity['gpu_uuid'])
        with (out / name).open('a') as stream:
            stream.write(json.dumps(dict(sample=index, gpu=snapshot)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        external = [mib for pid, mib in snapshot['processes'].items() if pid != identity['pid']]
        require(all(0 <= mib <= EXTERNAL_PER_PROCESS_MAX_MIB for mib in external)
                and sum(external) < EXTERNAL_TOTAL_MAX_MIB, 'GPU_DRAIN_EXTERNAL_RESOURCE_LOAD')
        if identity['pid'] not in snapshot['processes'] and snapshot['memory_mib'] < 1024:
            return snapshot
        if index < 80:
            time.sleep(.25)
    raise ValueError('HOLDER_GPU_CONTEXT_NOT_DRAINED')


def restore_holder(out, holder, identity):
    pane = holder['pane']
    for _ in range(80):
        require(pane_value(pane, '#{pane_id}') == holder['pane_id'], 'PANE_IDENTITY_CHANGED')
        status = proc_status(identity['pid'])
        require(status is None or status['state'] in ('Z', 'X'), 'HOLDER_NOT_TERMINAL')
        if pane_value(pane, '#{pane_dead}') == '1':
            break
        time.sleep(.25)
    require(pane_value(pane, '#{pane_dead}') == '1', 'PANE_DEATH_NOT_OBSERVED')
    # Own-phase leftovers (e.g. a lingering DataLoader worker context) may drain
    # slowly; give the card up to 180s before failing restoration.
    drain_deadline = time.monotonic() + 180
    last_error = None
    while time.monotonic() < drain_deadline:
        try:
            wait_gpu_drain(out, identity, 'GPU_RESTORE_DRAIN_%s.jsonl' % pane_value(pane, '#{pane_id}'))
            last_error = None
            break
        except ValueError as exc:
            if 'EXTERNAL_RESOURCE_LOAD' not in str(exc):
                raise
            last_error = exc
            time.sleep(2)
    if last_error is not None:
        raise last_error
    call('tmux', 'respawn-pane', '-t', pane, '-c', str(ROOT), holder['command'])
    restored_pid = int(pane_value(pane, '#{pane_pid}'))
    observed = None
    snapshot = None
    for _ in range(60):
        time.sleep(.5)
        if not (Path('/proc') / str(restored_pid)).exists():
            break
        observed = proc(restored_pid)
        require(observed['command'] == holder['command'] and observed['cwd'] == str(ROOT)
                and observed['proc_uid'] == identity['proc_uid'], 'RESTORED_PROCESS_IDENTITY')
        snapshot = gpu_snapshot(holder['gpu_uuid'])
        if snapshot['processes'].get(restored_pid, 0) > HOLDER_MIN_MIB:
            return dict(restored=True, pid=restored_pid, process_identity=observed, gpu=snapshot)
    return dict(restored=False, pid=restored_pid, process_identity=observed, gpu=snapshot,
                manual_recovery_required=True)


def run_step(step, out, deadline):
    log = (out / ('step_%s.log' % step['name'])).open('w')
    env = dict(os.environ)
    env.update(step.get('env_extra', {}))
    require(step['argv'][0].endswith('python3') or 'torchrun' in step['argv'][0]
            or step['argv'][0].endswith('python'), 'STEP_INTERPRETER')
    require(any(str(HERE) in str(a) for a in step['argv']), 'STEP_SCRIPT_SCOPE')
    process = subprocess.Popen(step['argv'], env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=str(HERE))
    try:
        while True:
            code = process.poll()
            if code is not None:
                require(code == 0, 'STEP_%s_EXIT_%d' % (step['name'].upper(), code))
                return dict(name=step['name'], exit=0)
            require(time.monotonic() < deadline, 'STEP_DEADLINE:' + step['name'])
            time.sleep(2)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    finally:
        log.close()


def main():
    runbook_path = Path(sys.argv[1]).resolve()
    require(runbook_path.parent == HERE, 'RUNBOOK_SCOPE')
    runbook = json.loads(runbook_path.read_text())
    require(runbook['runbook_sha256_self'] == 'SELF_EXCLUDED', 'RUNBOOK_FORMAT')
    for name, digest in runbook['code_sha256'].items():
        require(sha256(HERE / name) == digest, 'CODE_CHANGED:' + name)
    out = HERE / runbook['out']
    out.mkdir(parents=True, exist_ok=False)
    holders = runbook['holders']
    require(1 <= len(holders) <= 4, 'HOLDER_COUNT')
    identities = [holder_identity(h) for h in holders]
    for identity, holder in zip(identities, holders):
        snapshot = gpu_snapshot(holder['gpu_uuid'])
        require(identity['pid'] in snapshot['processes'], 'HOLDER_GPU_IDENTITY')
        require(snapshot['processes'][identity['pid']] > HOLDER_MIN_MIB, 'HOLDER_NOT_OCCUPYING')
        external = [mib for pid, mib in snapshot['processes'].items() if pid != identity['pid']]
        require(all(0 <= mib <= EXTERNAL_PER_PROCESS_MAX_MIB for mib in external)
                and sum(external) <= EXTERNAL_TOTAL_MAX_MIB, 'EXTERNAL_RESOURCE_LOAD')
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        raise InterruptedError('LEASE_SIGNAL:%d' % signum)

    for s in original:
        signal.signal(s, interrupted)
    attempted = False
    error = None
    restorations = []
    success = False
    lease_started = None
    try:
        save(out / 'LEASE_BEFORE.json', dict(identities=identities, runbook=runbook['name']))
        for holder in holders:
            call('tmux', 'set-option', '-w', '-t', holder['pane'], 'remain-on-exit', 'on')
        for identity, holder in zip(identities, holders):
            require(identity_equal(holder_identity(holder), identity), 'IDENTITY_CHANGED_PRE_SIGNAL')
        attempted = True
        lease_started = time.monotonic()
        for identity in identities:
            require(identity_equal(holder_identity(next(
                h for h in holders if h['pane'] == identity['pane'])), identity),
                'IDENTITY_CHANGED_FINAL')
            os.kill(identity['pid'], signal.SIGTERM)
        for identity in identities:
            wait_holder_terminal(identity)
        for identity in identities:
            wait_gpu_drain(out, identity, 'GPU_LEASE_DRAIN_%s.jsonl' % identity['pane_id'])
        save(out / 'LEASE_ACTIVE.json', dict(holders_exited=True, unix=time.time()))
        deadline = lease_started + runbook['lease_wall_seconds'] - 120
        for step in runbook['steps']:
            run_step(step, out, deadline)
        require(time.monotonic() < lease_started + runbook['lease_wall_seconds'], 'LEASE_WALL')
        success = True
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        for s in original:
            signal.signal(s, signal.SIG_IGN)
        try:
            if not attempted:
                restorations = [dict(restored=True, not_borrowed=True) for _ in holders]
            else:
                for holder, identity in zip(holders, identities):
                    status = proc_status(identity['pid'])
                    retained = False
                    if status is not None and status['state'] not in ('Z', 'X'):
                        try:
                            retained = identity_equal(holder_identity(holder), identity)
                        except FileNotFoundError:
                            retained = False
                        if not retained:
                            wait_holder_terminal(identity)
                    if retained:
                        snapshot = gpu_snapshot(holder['gpu_uuid'])
                        require(snapshot['processes'].get(identity['pid'], 0) > HOLDER_MIN_MIB,
                                'ORIGINAL_HOLDER_RETENTION_NOT_OBSERVED')
                        restorations.append(dict(restored=True, original_holder_retained=True,
                                                 pid=identity['pid']))
                    else:
                        restorations.append(restore_holder(out, holder, identity))
        except BaseException as exc:
            restorations.append(dict(restored=False, error=repr(exc), manual_recovery_required=True))
        finally:
            for holder, restoration in zip(holders, restorations):
                try:
                    require(pane_value(holder['pane'], '#{pane_id}') == holder['pane_id'],
                            'PANE_CHANGED')
                    if restoration.get('restored') and pane_value(holder['pane'], '#{pane_dead}') == '0':
                        call('tmux', 'set-option', '-w', '-t', holder['pane'], 'remain-on-exit', 'off')
                    else:
                        call('tmux', 'set-option', '-w', '-t', holder['pane'], 'remain-on-exit', 'on')
                except BaseException as exc:
                    restoration['restored'] = False
                    restoration['option_restore_error'] = repr(exc)
            save(out / 'RESTORATION.json', dict(holders=restorations))
            save(out / 'LEASE_RESULT.json', dict(
                execute_returned=success, error=error, holder_signal_attempted=attempted,
                holders_restored=all(r.get('restored') for r in restorations),
                external_processes_stopped=0, scientific_pass=False))
            for s, handler in original.items():
                signal.signal(s, handler)
    if error or not success or not all(r.get('restored') for r in restorations):
        raise RuntimeError('LEASE_FAILED:' + str(error))
    print(json.dumps(dict(status='LEASE_COMPLETED', runbook=runbook['name'])))


if __name__ == '__main__':
    main()
