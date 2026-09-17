"""Bounded persistent production queues. Child transports own GPU cleanup.

Fresh runs only; never retry a failed or interrupted job. SIGTERM/SIGINT drain
the current bounded transport, then stop. No GPU process is signalled here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LINE = ROOT / 'projects/qwen35_indoor_nav'


def scoped(path):
    p = Path(path)
    if not p.is_absolute() or p != p.resolve() or not p.is_relative_to(LINE):
        raise ValueError('EXACT_PROJECT_LINE_PATH_REQUIRED')
    return p


def digest(path):
    h = hashlib.sha256()
    with scoped(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(scoped(path).read_text())


def write_new(path, value):
    with scoped(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with scoped(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def field(value, key):
    for part in key.split('.'):
        value = value[part]
    return value


def receipts(checks):
    evidence = {}
    for check in checks:
        path = scoped(check['path'])
        data = read(path)
        for key, expected in check['equals'].items():
            actual = field(data, key)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError('COMPLETION_RECEIPT_FAILED:' + key)
        evidence[str(path)] = digest(path)
    return evidence


def check_command(command):
    if not isinstance(command, list) or len(command) < 4:
        raise ValueError('EXPLICIT_PYTHON_ARGV_REQUIRED')
    # Project-local env python may itself be a registered symlink. Resolve it
    # inside ROOT, while scripts must be non-symlink paths inside this line.
    python = Path(command[0])
    if not python.is_absolute() or not python.resolve().is_relative_to(ROOT):
        raise ValueError('PROJECT_PYTHON_REQUIRED')
    if command[1:3] != ['-I', '-B']:
        raise ValueError('ISOLATED_NO_BYTECODE_REQUIRED')
    scoped(command[3])
    if not Path(command[3]).is_file():
        raise ValueError('MISSING_SCRIPT')


def validate(plan):
    if plan['training_allowed'] is not False or plan['automatic_retry'] is not False:
        raise ValueError('DATA_ONLY_NO_RETRY')
    if not 1 <= plan['wall_seconds'] <= 86400:
        raise ValueError('BOUNDED_QUEUE_REQUIRED')
    jobs = plan['jobs']
    if not jobs or len(jobs) > 64 or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('UNIQUE_BOUNDED_JOBS_REQUIRED')
    all_checks = set()
    for job in jobs:
        if not isinstance(job['id'], str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', job['id']):
            raise ValueError('SAFE_JOB_ID_REQUIRED')
        if type(job['gpu']) is not int or job['gpu'] not in range(1, 8):
            raise ValueError('NO_GPU0')
        if not 1 <= job['max_seconds'] <= plan['wall_seconds']:
            raise ValueError('JOB_CLEANUP_BUDGET_REQUIRED')
        transport_seconds = job.get('transport_upper_seconds')
        if type(transport_seconds) is not int or transport_seconds < 1:
            raise ValueError('ORIGINAL_TRANSPORT_UPPER_BOUND_REQUIRED')
        if not job['completion']:
            raise ValueError('COMPLETION_EVIDENCE_REQUIRED')
        check_command(job['command'])
        if job.get('audit_command'):
            check_command(job['audit_command'])
            seconds = job.get('audit_seconds', 1200)
            if type(seconds) is not int or not 1 <= seconds <= 1800 or seconds >= job['max_seconds']:
                raise ValueError('BOUNDED_AUDIT_WITHIN_JOB_RESERVATION')
        else:
            seconds = 0
        if transport_seconds + seconds > job['max_seconds']:
            raise ValueError('TRANSPORT_PLUS_AUDIT_EXCEEDS_RESERVATION')
        for check in job['completion']:
            path = str(scoped(check['path']))
            if path in all_checks or not check['equals']:
                raise ValueError('DISJOINT_NONEMPTY_COMPLETION_REQUIRED')
            all_checks.add(path)
        if not job['input_hashes']:
            raise ValueError('FROZEN_INPUTS_REQUIRED')
        for path in job['input_hashes']:
            scoped(path)
        for command in [job['command']] + ([job['audit_command']] if job.get('audit_command') else []):
            if command[3] not in job['input_hashes']:
                raise ValueError('ENTRY_NOT_FROZEN')


def verify_job(job):
    for path, expected in job['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError('FROZEN_INPUT_CHANGED:' + path)
    if any(scoped(c['path']).exists() for c in job['completion']):
        raise ValueError('OLD_ATTEMPT_NOT_RESTARTED')


def ready_to_start(job, now, deadline, stop):
    return not stop and now + job['max_seconds'] <= deadline


def execute(plan_path, gpu):
    plan_path = scoped(plan_path)
    plan = read(plan_path)
    validate(plan)
    approval = read(plan_path.parent / 'MAIN_AGENT_APPROVAL.json')
    if approval.get('approved') is not True or approval['plan_sha256'] != digest(plan_path):
        raise ValueError('MAIN_APPROVAL_REQUIRED')
    if approval['queue_source_sha256'] != digest(HERE / 'queue.py'):
        raise ValueError('APPROVED_QUEUE_SOURCE_CHANGED')
    jobs = [j for j in plan['jobs'] if j['gpu'] == gpu]
    if not jobs:
        raise ValueError('GPU_NOT_IN_APPROVED_QUEUE')
    locks = HERE / 'gpu_locks'
    locks.mkdir(exist_ok=True)
    lease = scoped(locks / f'gpu_{gpu}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = plan_path.parent / f'lane_gpu_{gpu}'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + plan['wall_seconds']
    stopping = [False]
    def drain(signum, _frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    write_new(out / 'IDENTITY.json', dict(pid=os.getpid(), gpu=gpu,
        started_unix=time.time(), deadline_monotonic=deadline,
        plan_sha256=digest(plan_path), cleanup_owner='original_child_transport',
        signal_behavior='drain_current_child_then_stop_no_gpu_signals'))
    events = out / 'EVENTS.jsonl'
    states = {j['id']: 'PENDING' for j in jobs}
    error = None
    child = None
    try:
        for job in jobs:
            if not ready_to_start(job, time.monotonic(), deadline, stopping[0]):
                append(events, dict(event='queue_draining_or_deadline', unix=time.time()))
                break
            verify_job(job)
            if not ready_to_start(job, time.monotonic(), deadline, stopping[0]):
                break
            # Durable attempt reservation precedes Popen. A crash cannot silently
            # retry: this lane/output can only be created once.
            write_new(out / (job['id'] + '_RESERVED.json'), dict(job=job, unix=time.time()))
            states[job['id']] = 'RUNNING'
            append(events, dict(event='production_start', job=job['id'], unix=time.time()))
            with (out / (job['id'] + '_production.log')).open('x') as log:
                if not ready_to_start(job, time.monotonic(), deadline, stopping[0]):
                    states[job['id']] = 'RESERVED_NOT_LAUNCHED_DRAIN_OR_DEADLINE'
                    break
                child = subprocess.Popen(job['command'], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                append(events, dict(event='child_started', job=job['id'], pid=child.pid, unix=time.time()))
                # No outer kill: the registered original supervisor enforces its
                # own wall/resource budget and finally restores borrowed holders.
                rc = child.wait()
                child = None
            if rc != 0:
                raise RuntimeError('PRODUCTION_NONZERO_NO_RETRY:' + str(rc))
            evidence = receipts(job['completion'])
            states[job['id']] = 'PRODUCED_PENDING_AUDIT'
            if job.get('audit_command'):
                append(events, dict(event='audit_start', job=job['id'], unix=time.time()))
                with (out / (job['id'] + '_audit.log')).open('x') as log:
                    audit = subprocess.run(job['audit_command'], cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, timeout=job.get('audit_seconds', 1200), check=False)
                if audit.returncode != 0:
                    raise RuntimeError('AUDIT_NONZERO_NO_RETRY:' + str(audit.returncode))
                if receipts(job['completion']) != evidence:
                    raise RuntimeError('RECEIPTS_CHANGED_DURING_AUDIT')
                states[job['id']] = 'AUDITED_SEE_PER_ITEM_QUALITY'
            append(events, dict(event='job_closed', job=job['id'], state=states[job['id']],
                                evidence=evidence, unix=time.time()))
    except BaseException as exc:
        error = repr(exc)
        for key, value in states.items():
            if value in ('RUNNING', 'PRODUCED_PENDING_AUDIT'):
                states[key] = 'FAILED_NO_AUTOMATIC_RETRY'
        append(events, dict(event='lane_stopped', error=error, unix=time.time()))
    finally:
        try:
            # An event/fsync failure after Popen must never release the lease
            # while its producer is still alive. Child transport owns cleanup.
            if child is not None:
                child.wait()
            write_new(out / 'RESULT.json', dict(states=states, error=error,
                elapsed_seconds=time.monotonic()-started, training_started=False,
                scientific_pass=False, automatic_retries=0, gpu_processes_signalled=0))
        finally:
            lease.close()
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    execute(Path(args.plan), args.gpu)
