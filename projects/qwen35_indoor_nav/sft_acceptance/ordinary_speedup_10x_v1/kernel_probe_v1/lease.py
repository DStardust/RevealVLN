"""GPU3 exact-holder lease for the bounded kernel probe. Stdlib only.

Pattern follows the approved batch_execution_v1/gpu5_transport_v2/transport.py:
live identity re-read from the exact tmux pane, double /proc identity check,
SIGTERM to the exact holder PID only, GPU drain evidence, own-child cleanup,
finally restore via tmux respawn-pane with the same argv/cwd, restoration
verification by live identity plus GPU re-occupancy. No PID from any old
document is trusted; identity is re-read at execution time.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]

GPU_INDEX = 3
GPU_UUID = 'GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a'
PANE = 'vla_idle_occupancy_20260904:0.0'
PANE_ID = '%153'
HOLDER_COMMAND = ('.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu 3 '
                  '--reserve-mib 2048 --max-initial-used-mib 1536 --tag vla_idle_occupancy_20260904')
HOLDER_MIN_MIB = 20000
EXTERNAL_PER_PROCESS_MAX_MIB = 768
EXTERNAL_TOTAL_MAX_MIB = 1024
LEASE_WALL_SECONDS = 1200
ENV_PYTHON = LINE / '.envs/q35n_qwen_g2_v1/bin/python3'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def call(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def pane(fmt):
    return call('tmux', 'display-message', '-p', '-t', PANE, fmt)


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


def gpu_snapshot():
    out = subprocess.check_output(
        ['nvidia-smi', '-i', GPU_UUID, '--query-compute-apps=pid,used_memory',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = {}
    for line in out.strip().splitlines():
        if line.strip():
            pid, mib = [x.strip() for x in line.split(',')]
            processes[int(pid)] = int(mib)
    used = subprocess.check_output(
        ['nvidia-smi', '-i', GPU_UUID, '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    return dict(uuid=GPU_UUID, processes=processes, memory_mib=int(used))


def holder_identity():
    require(pane('#{pane_id}') == PANE_ID, 'PANE_IDENTITY_CHANGED')
    require(pane('#{pane_dead}') == '0', 'HOLDER_PANE_DEAD')
    require(pane('#{pane_current_path}') == str(ROOT), 'PANE_CWD')
    pid = int(pane('#{pane_pid}'))
    identity = proc(pid)
    require(identity['command'] == HOLDER_COMMAND, 'HOLDER_COMMAND_MISMATCH')
    require(identity['cwd'] == str(ROOT) and identity['proc_uid'] == 0, 'HOLDER_CWD_UID_MISMATCH')
    identity.update(pane=PANE, pane_id=PANE_ID)
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
        snapshot = gpu_snapshot()
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
    raise ValueError('HOLDER_GPU_CONTEXT_NOT_DRAINED_WITHIN_20_SECONDS')


def run_probe(mode, run_dir, deadline, env_extra):
    log = (run_dir / ('probe_%s.log' % mode)).open('w')
    env = dict(os.environ)
    env.update(env_extra)
    process = subprocess.Popen(
        [str(ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'probe.py'), '--mode', mode, '--out', str(run_dir)],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            code = process.poll()
            if code is not None:
                require(code == 0, 'PROBE_%s_EXIT_%d' % (mode.upper(), code))
                return
            require(time.monotonic() < deadline, 'PROBE_WALL_DEADLINE')
            time.sleep(2)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    finally:
        log.close()


def restore_dead_holder(out, identity):
    for _ in range(80):
        require(pane('#{pane_id}') == PANE_ID, 'PANE_IDENTITY_CHANGED')
        status = proc_status(identity['pid'])
        require(status is None or status['state'] in ('Z', 'X'), 'HOLDER_NOT_TERMINAL')
        if pane('#{pane_dead}') == '1':
            break
        time.sleep(.25)
    require(pane('#{pane_dead}') == '1', 'PANE_DEATH_NOT_OBSERVED')
    snapshot = wait_gpu_drain(out, identity, 'GPU_RESTORE_DRAIN.jsonl')
    save(out / 'GPU_PRE_RESTORE.json', snapshot)
    call('tmux', 'respawn-pane', '-t', PANE, '-c', str(ROOT), HOLDER_COMMAND)
    restored_pid = int(pane('#{pane_pid}'))
    observed = None
    for _ in range(60):
        time.sleep(.5)
        if not (Path('/proc') / str(restored_pid)).exists():
            break
        observed = proc(restored_pid)
        require(observed['command'] == HOLDER_COMMAND and observed['cwd'] == str(ROOT)
                and observed['proc_uid'] == identity['proc_uid'], 'RESTORED_PROCESS_IDENTITY')
        snapshot = gpu_snapshot()
        if snapshot['processes'].get(restored_pid, 0) > HOLDER_MIN_MIB:
            return dict(restored=True, pid=restored_pid, process_identity=observed, gpu=snapshot)
    return dict(restored=False, pid=restored_pid, process_identity=observed,
                gpu=gpu_snapshot(), manual_recovery_required=True)


def main():
    run_dir = HERE / 'run_v1'
    run_dir.mkdir(exist_ok=False)
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    for name, digest in protocol['code_sha256'].items():
        target = HERE / name
        require(target.is_file(), 'MISSING_LOCKED_CODE:' + name)
        require(__import__('hashlib').sha256(target.read_bytes()).hexdigest() == digest,
                'CODE_CHANGED:' + name)
    require(protocol['gpu']['uuid'] == GPU_UUID and protocol['gpu']['pane'] == PANE
            and protocol['gpu']['pane_id'] == PANE_ID
            and protocol['gpu']['holder_command'] == HOLDER_COMMAND, 'PROTOCOL_GPU_SCOPE')
    require(protocol['budget']['lease_wall_seconds'] == LEASE_WALL_SECONDS
            and protocol['budget']['max_forward_decisions'] == 512
            and protocol['budget']['max_probe_optimizer_updates'] == 8
            and protocol['budget']['max_own_gpu_memory_bytes'] == 30064771072, 'PROTOCOL_BUDGET')

    identity = holder_identity()
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        raise InterruptedError('LEASE_SIGNAL:%d' % signum)

    for s in original:
        signal.signal(s, interrupted)
    attempted = False
    remain = None
    remain_changed = False
    error = None
    restoration = None
    success = False
    try:
        before = gpu_snapshot()
        require(before['uuid'] == GPU_UUID and identity['pid'] in before['processes'], 'HOLDER_GPU_IDENTITY')
        require(before['processes'][identity['pid']] > HOLDER_MIN_MIB, 'HOLDER_NOT_OCCUPYING_EXPECTED_GPU')
        external = [mib for pid, mib in before['processes'].items() if pid != identity['pid']]
        require(all(0 <= mib <= EXTERNAL_PER_PROCESS_MAX_MIB for mib in external)
                and sum(external) <= EXTERNAL_TOTAL_MAX_MIB, 'EXTERNAL_RESOURCE_LOAD')
        remain = call('tmux', 'show-options', '-w', '-v', '-t', PANE, 'remain-on-exit')
        require(remain in ('on', 'off', 'failed'), 'TMUX_REMAIN_VALUE')
        save(run_dir / 'LEASE_BEFORE.json', dict(
            identity=identity, gpu=before, remain_on_exit=remain,
            restore_argv=['tmux', 'respawn-pane', '-t', PANE, '-c', str(ROOT), HOLDER_COMMAND],
            restore_no_force_kill=True, pidfd_available=False,
            residual_race='double /proc identity checks then exact PID SIGTERM; no atomic pidfd'))
        remain_changed = True
        call('tmux', 'set-option', '-w', '-t', PANE, 'remain-on-exit', 'on')
        require(identity_equal(holder_identity(), identity), 'HOLDER_IDENTITY_CHANGED_BEFORE_SIGNAL')
        require(identity_equal(holder_identity(), identity), 'HOLDER_IDENTITY_CHANGED_FINAL_CHECK')
        attempted = True
        os.kill(identity['pid'], signal.SIGTERM)
        lease_started = time.monotonic()
        wait_holder_terminal(identity)
        wait_gpu_drain(run_dir, identity, 'GPU_LEASE_DRAIN.jsonl')
        save(run_dir / 'LEASE_ACTIVE.json', dict(holder_exited=True, pane=PANE, pane_id=PANE_ID,
                                                 unix=time.time()))
        env = dict(CUDA_VISIBLE_DEVICES=GPU_UUID, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        deadline = lease_started + LEASE_WALL_SECONDS - 60
        run_probe('reference', run_dir, deadline, env)
        run_probe('candidate', run_dir, deadline, env)
        require(time.monotonic() < lease_started + LEASE_WALL_SECONDS, 'LEASE_WALL_EXCEEDED')
        success = True
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        for s in original:
            signal.signal(s, signal.SIG_IGN)
        try:
            if not attempted:
                restoration = dict(restored=True, not_borrowed=True)
            else:
                status = proc_status(identity['pid'])
                retained = False
                if status is not None and status['state'] not in ('Z', 'X'):
                    try:
                        retained = identity_equal(holder_identity(), identity)
                    except FileNotFoundError:
                        retained = False
                    if not retained:
                        wait_holder_terminal(identity)
                if retained:
                    snapshot = gpu_snapshot()
                    require(snapshot['processes'].get(identity['pid'], 0) > HOLDER_MIN_MIB,
                            'ORIGINAL_HOLDER_RETENTION_NOT_OBSERVED')
                    restoration = dict(restored=True, original_holder_retained=True, pid=identity['pid'])
                else:
                    restoration = restore_dead_holder(run_dir, identity)
        except BaseException as exc:
            restoration = dict(restored=False, error=type(exc).__name__ + ': ' + str(exc),
                               manual_recovery_required=True, unknown_processes_stopped=0)
        finally:
            if remain_changed:
                try:
                    require(pane('#{pane_id}') == PANE_ID, 'PANE_CHANGED_NO_OPTION_RESTORE')
                    if restoration.get('restored') and pane('#{pane_dead}') == '0':
                        call('tmux', 'set-option', '-w', '-t', PANE, 'remain-on-exit', remain)
                        restoration['remain_on_exit_restored'] = True
                    else:
                        call('tmux', 'set-option', '-w', '-t', PANE, 'remain-on-exit', 'on')
                        restoration['remain_on_exit_preserved_on_for_recovery'] = True
                except BaseException as exc:
                    restoration['restored'] = False
                    restoration['option_restore_error'] = repr(exc)
            try:
                save(run_dir / 'RESTORATION.json', restoration)
                save(run_dir / 'LEASE_RESULT.json', dict(
                    execute_returned=success, error=error, holder_signal_attempted=attempted,
                    holder_restored=restoration['restored'], external_processes_stopped=0,
                    scientific_pass=False, probe_is_not_training_or_navigation=True))
            finally:
                for s, handler in original.items():
                    signal.signal(s, handler)
    if error or not success or not restoration['restored']:
        raise RuntimeError('LEASE_OR_PROBE_FAILED:' + str(error))
    print(json.dumps(dict(status='LEASE_COMPLETED', restoration=restoration), allow_nan=False))


if __name__ == '__main__':
    main()
