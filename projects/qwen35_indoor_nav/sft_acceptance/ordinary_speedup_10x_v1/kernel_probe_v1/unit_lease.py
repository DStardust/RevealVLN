"""One-shot GPU3 lease for unit_kernel_check.py; reuses lease.py primitives.

Same discipline as the registered probe: live identity, double check, exact
SIGTERM, drain evidence, finally restore with same argv/cwd, verification.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location('kernel_probe_lease', HERE / 'lease.py')
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)

UNIT_SECONDS = 600


def main():
    run_dir = HERE / 'run_unit_v1'
    run_dir.mkdir(exist_ok=True)
    identity = lease.holder_identity()
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        raise InterruptedError('UNIT_LEASE_SIGNAL:%d' % signum)

    for s in original:
        signal.signal(s, interrupted)
    attempted = False
    error = None
    restoration = None
    success = False
    try:
        before = lease.gpu_snapshot()
        lease.require(before['uuid'] == lease.GPU_UUID
                      and identity['pid'] in before['processes'], 'HOLDER_GPU_IDENTITY')
        lease.require(before['processes'][identity['pid']] > lease.HOLDER_MIN_MIB,
                      'HOLDER_NOT_OCCUPYING_EXPECTED_GPU')
        external = [mib for pid, mib in before['processes'].items() if pid != identity['pid']]
        lease.require(all(0 <= mib <= lease.EXTERNAL_PER_PROCESS_MAX_MIB for mib in external)
                      and sum(external) <= lease.EXTERNAL_TOTAL_MAX_MIB, 'EXTERNAL_RESOURCE_LOAD')
        remain = lease.call('tmux', 'show-options', '-w', '-v', '-t', lease.PANE, 'remain-on-exit')
        lease.call('tmux', 'set-option', '-w', '-t', lease.PANE, 'remain-on-exit', 'on')
        lease.require(lease.identity_equal(lease.holder_identity(), identity), 'IDENTITY_BEFORE_SIGNAL')
        lease.require(lease.identity_equal(lease.holder_identity(), identity), 'IDENTITY_FINAL_CHECK')
        attempted = True
        os.kill(identity['pid'], signal.SIGTERM)
        lease.wait_holder_terminal(identity)
        lease.wait_gpu_drain(run_dir, identity, 'UNIT_GPU_LEASE_DRAIN.jsonl')
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=lease.GPU_UUID, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log = (run_dir / 'unit_kernel_check.log').open('w')
        process = subprocess.Popen(
            [str(lease.ENV_PYTHON), '-I', '-B', '-u', str(HERE / 'unit_kernel_check.py')],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + UNIT_SECONDS
        try:
            while True:
                code = process.poll()
                if code is not None:
                    lease.require(code == 0, 'UNIT_CHECK_EXIT_%d' % code)
                    break
                lease.require(time.monotonic() < deadline, 'UNIT_CHECK_DEADLINE')
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
                status = lease.proc_status(identity['pid'])
                retained = False
                if status is not None and status['state'] not in ('Z', 'X'):
                    try:
                        retained = lease.identity_equal(lease.holder_identity(), identity)
                    except FileNotFoundError:
                        retained = False
                    if not retained:
                        lease.wait_holder_terminal(identity)
                if retained:
                    restoration = dict(restored=True, original_holder_retained=True, pid=identity['pid'])
                else:
                    restoration = lease.restore_dead_holder(run_dir, identity)
        except BaseException as exc:
            restoration = dict(restored=False, error=repr(exc), manual_recovery_required=True)
        finally:
            try:
                lease.require(lease.pane('#{pane_id}') == lease.PANE_ID, 'PANE_CHANGED')
                if restoration.get('restored') and lease.pane('#{pane_dead}') == '0':
                    lease.call('tmux', 'set-option', '-w', '-t', lease.PANE, 'remain-on-exit', 'off')
            except BaseException as exc:
                restoration['restored'] = False
                restoration['option_restore_error'] = repr(exc)
            lease.save(run_dir / 'UNIT_RESTORATION.json', restoration)
            lease.save(run_dir / 'UNIT_LEASE_RESULT.json', dict(
                execute_returned=success, error=error, holder_signal_attempted=attempted,
                holder_restored=restoration['restored'], external_processes_stopped=0))
            for s, handler in original.items():
                signal.signal(s, handler)
    if error or not success or not restoration['restored']:
        raise RuntimeError('UNIT_LEASE_FAILED:' + str(error))
    print(json.dumps(dict(status='UNIT_LEASE_COMPLETED')))


if __name__ == '__main__':
    main()
