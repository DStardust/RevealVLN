"""Independent bounded recapture after the already-owned evaluation completes."""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u
from transfer_pipeline import placeholder
from unseen_pipeline import devices
from contracts import summarize_coverage


def records(run, sealed=True):
    result = {}
    for session in sorted((run / 'capture').glob('*')):
        seal_path = session / 'STATE_SEAL.json'
        if sealed and not seal_path.exists():
            continue
        seal = u.read(seal_path) if seal_path.exists() else None
        if sealed:
            if seal['base_before'] != seal['base_after'] or seal['source_lock_sha256'] != u.sha(run / 'SOURCE_LOCK.json'):
                raise ValueError('INVALID_CAPTURE_STATE_SEAL')
        for path in session.glob('episodes/*/COMPLETE.json'):
            row = u.read(path)
            if sealed and row['trajectory_id'] not in seal['complete_ids']:
                continue
            if row['trajectory_id'] in result:
                raise ValueError('DUPLICATE_CAPTURE_GROUP')
            result[row['trajectory_id']] = dict(row, receipt_path=str(path))
    return result


def process_start(pid):
    # /proc comm may contain spaces; the starttime is field 22.
    text = Path(f'/proc/{pid}/stat').read_text()
    return text[text.rfind(')') + 2:].split()[19]


def owned(worker):
    proc = worker['proc']
    if proc.poll() is not None:
        return False
    try:
        return os.getpgid(proc.pid) == proc.pid and process_start(proc.pid) == worker['start_ticks']
    except ProcessLookupError:
        return False
    except FileNotFoundError:
        return False


def account_interrupted_attempts(run, now):
    for path in (run / 'attempts').glob('*.start.json'):
        record = u.read(path)
        finished = path.with_name(path.name.replace('.start.json', '.json'))
        if finished.exists():
            continue
        try:
            still_same = process_start(record['pid']) == record['start_ticks']
        except (FileNotFoundError, ProcessLookupError):
            still_same = False
        if still_same:
            raise RuntimeError('PREVIOUS_CAPTURE_WORKER_STILL_RUNNING')
        u.write(finished, dict(record, returncode=None,
            wall_seconds=max(0., now - record['started_unix']),
            duration_kind='UPPER_BOUND_INCLUDES_POSSIBLE_DOWNTIME',
            reason='Launcher ended before receipt; actual exit time unavailable, never charged as zero.'))


def main(run, resume):
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    u.verify_sources(run)
    p = u.read(run / 'PROTOCOL.json')
    started_file = run / 'STARTED.json'
    if started_file.exists() and not resume:
        raise ValueError('RESUME_REQUIRED')
    if (run / 'RESULT.json').exists() and u.read(run / 'RESULT.json')['status'] == 'COMPLETE':
        raise ValueError('CLOSED_RUN')
    if not started_file.exists():
        u.write(started_file, dict(unix=time.time(), pid=os.getpid()))
    for folder in ('attempts', 'capture', 'failed_attempts', 'sessions'):
        (run / folder).mkdir(exist_ok=True)
    # A hard interruption can leave an unsealed session. Preserve it, never splice it.
    for session in (run / 'capture').iterdir():
        if not (session / 'STATE_SEAL.json').exists():
            session.rename(run / 'failed_attempts' / (session.name + '_' + str(time.time_ns())))
    plans = u.read(run / 'REQUESTS.json')
    account_interrupted_attempts(run, time.time())
    prior_attempts = [u.read(path) for path in (run / 'attempts').glob('*.json') if not path.name.endswith('.start.json')]
    hours = sum(row['wall_seconds'] / 3600 for row in prior_attempts)
    workers = []
    leased = False
    began = time.time()
    capture_started = None
    state = dict(status='RUNNING', phase='WAIT_DEPENDENCY', planned_trajectories=p['planned_trajectories'],
        planned_missing=p['planned_missing'], planned_stop=p['planned_stop'], dependency=p['wait_for_service'],
        unknown_attempt_durations=sum(row.get('duration_kind') == 'UPPER_BOUND_INCLUDES_POSSIBLE_DOWNTIME'
                                      for row in prior_attempts))
    def save():
        complete = records(run)
        coverage = summarize_coverage(plans, list(complete.values()))
        state.update(unix=time.time(), recorded_trajectories=len(records(run, False)),
            sealed_trajectories=len(complete),
            captured_missing=coverage['totals']['captured_missing_actor_positions'],
            captured_stop=coverage['totals']['captured_missing_stop_positions'],
            gpu_hours=hours + sum((time.time() - w['start']) / 3600 for w in workers),
            workers=[dict(pid=w['proc'].pid, gpu=w['gpu'], uuid=w['uuid'], output=str(w['out']),
                progress=u.read(w['out'] / 'PROGRESS.json') if (w['out'] / 'PROGRESS.json').exists() else None)
                for w in workers])
        u.write(run / 'STATUS.json', state)
    def interrupted(signum, frame):
        raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def finish(w):
        nonlocal hours
        elapsed = time.time() - w['start']; hours += elapsed / 3600
        w['log'].close()
        u.write(w['path'].with_suffix('.json'), dict(pid=w['proc'].pid, gpu=w['gpu'], uuid=w['uuid'],
            start_ticks=w['start_ticks'], wall_seconds=elapsed, returncode=w['proc'].returncode,
            output=str(w['out']), ids=w['ids'], duration_kind='OBSERVED_PROCESS_SESSION'))
    def spawn(gpu, ids):
        inventory = devices()
        if inventory[gpu]['free_mib'] < 26000:
            raise RuntimeError('INSUFFICIENT_MEMORY_FOR_REGISTERED_WORKER')
        name = f'gpu{gpu}_{time.time_ns()}'
        out = run / 'capture' / name
        path = run / 'attempts' / (name + '.log')
        log = path.open('x')
        uuid = inventory[gpu]['uuid']
        command = [str(u.PYTHON), '-I', '-B', str(HERE / 'worker.py'),
                   '--run', str(run), '--output', str(out), '--ids', ','.join(map(str, ids))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        proc = subprocess.Popen(command, cwd=u.REPO, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        w = dict(proc=proc, start_ticks=process_start(proc.pid), start=time.time(),
            gpu=gpu, uuid=uuid, out=out, path=path, log=log, ids=ids)
        workers.append(w)
        u.write(path.with_suffix('.start.json'), dict(pid=proc.pid, start_ticks=w['start_ticks'],
            started_unix=w['start'], gpu=gpu, uuid=uuid, output=str(out), ids=ids))
    def wait(refill=None):
        last_size = 0
        while workers:
            for w in list(workers):
                if w['proc'].poll() is not None:
                    finish(w); workers.remove(w)
                    if w['proc'].returncode:
                        raise RuntimeError('CAPTURE_FAILED:' + str(w['out']))
                    if refill:
                        refill(w['gpu'])
            save()
            if state['gpu_hours'] > p['gpu_session_hours_limit'] or time.time() - capture_started > p['capture_wall_hours_limit'] * 3600:
                raise RuntimeError('CAPTURE_BUDGET_REACHED')
            if time.time() - last_size > 60:
                state['artifact_bytes'] = sum(path.stat().st_size for path in run.rglob('*') if path.is_file())
                if state['artifact_bytes'] > p['artifact_limit_gib'] * 1024**3:
                    raise RuntimeError('CAPTURE_ARTIFACT_LIMIT')
                last_size = time.time()
            if workers:
                time.sleep(5)
    try:
        save()
        # Explicit dependency on one already-running owned job, not an idle-GPU poller.
        while True:
            dep = subprocess.run(['systemctl', 'show', p['wait_for_service'], '-p', 'SubState', '--value'],
                                 capture_output=True, text=True, timeout=10)
            if dep.returncode:
                raise RuntimeError('CANNOT_READ_OWNED_DEPENDENCY')
            substate = dep.stdout.strip()
            state['dependency_substate'] = substate
            if substate in ('exited', 'dead', 'failed'):
                break
            if substate not in ('running', 'start', 'start-post', 'stop', 'stop-sigterm', 'stop-sigkill'):
                raise RuntimeError('UNEXPECTED_DEPENDENCY_STATE:' + substate)
            if time.time() - u.read(started_file)['unix'] > p['wait_hours_limit'] * 3600:
                raise RuntimeError('DEPENDENCY_WAIT_BUDGET_REACHED')
            save(); time.sleep(30)
        before = placeholder('status')
        if before['leases'] or before['manual_paused'] or before['external_pids']:
            raise RuntimeError('RESOURCE_NOT_AVAILABLE_AFTER_DEPENDENCY')
        acquired = placeholder('acquire'); leased = True
        u.write(run / f'RESOURCE_LEASE_{time.time_ns()}.json', dict(before=before, acquired=acquired))
        inventory = devices()
        available = [g for g in p['gpu_indices'] if inventory[g]['free_mib'] >= 26000]
        u.write(run / f'GPU_INVENTORY_{time.time_ns()}.json', dict(devices=inventory, selected=available))
        if not available:
            raise RuntimeError('NO_AUTHORIZED_GPU_WITH_HEADROOM')
        capture_started = time.time()
        state['phase'] = 'GPU_SMOKE'
        done = records(run)
        smoke = [i for i in p['smoke_ids'] if i not in done]
        if smoke:
            spawn(available[0], smoke); wait()
        done = records(run)
        if not all(i in done for i in p['smoke_ids']):
            raise RuntimeError('SMOKE_NOT_SEALED')
        u.write(run / 'SMOKE_RESULT.json', dict(status='REAL_MODEL_AND_REPLAY_PASSED',
            ids=p['smoke_ids'], new_actor_positions=sum(done[i]['new_actor_positions'] for i in p['smoke_ids']),
            new_stop_positions=sum(done[i]['new_terminal_stop_positions'] for i in p['smoke_ids'])))
        state['phase'] = 'CAPTURE'
        pending = deque(i for i in p['run_order'] if i not in done)
        def refill(gpu):
            if pending:
                ids = [pending.popleft() for _ in range(min(p['chunk_trajectories'], len(pending)))]
                spawn(gpu, ids)
        for gpu in available:
            refill(gpu)
        wait(refill)
        state['phase'] = 'REVIEW'; save()
        done = records(run)
        coverage = summarize_coverage(plans, list(done.values()))
        assert coverage['status'] == 'COMPLETE' and len(done) == p['planned_trajectories'], 'INCOMPLETE_CAPTURE'
        for row in done.values():
            assert u.sha(row['cache_path']) == row['cache_sha256'], 'CAPTURE_CACHE_CHANGED'
            path = Path(row['receipt_path']).parent
            assert u.sha(path / 'TRACE.jsonl') == row['trace_sha256'], 'CAPTURE_TRACE_CHANGED'
        result = dict(status='COMPLETE', coverage=coverage, trajectories=len(done),
            actor_rows=sum(r['captured_all_actor_rows'] for r in done.values()),
            gpu_hours=hours, base_updates=0, optimizer_updates=0, navigation_sr_measured=False,
            new_independent_routes=0, training_admission=False,
            meaning='Actual all-token feature coverage; training/deployment all-token residual is a later common change.')
        u.write(run / 'RESULT.json', result)
        u.write(run / 'CAPTURE_INDEX.json', dict(rows=[done[k] for k in sorted(done)]))
        (run / 'REPORT_ZH.md').write_text('COMPLETE\n\n334条原轨迹全部真实回放，补齐21,396个缺失actor位置，其中133个STOP。'
            'FIT/DEV原分割保留；STOP不产生新观测，chunk使用原query_start记忆。\n\n'
            '这是特征覆盖修补，不增加独立路线数，未训练、未改变上线策略、未测新增SR。\n')
        state.update(status='COMPLETE', phase='REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, InterruptedError) else 'FAILED', error=str(error))
        u.write(run / f'FAILURE_{time.time_ns()}.json', dict(state, traceback=traceback.format_exc()))
        raise
    finally:
        for w in workers:
            if owned(w):
                os.killpg(w['proc'].pid, signal.SIGTERM)
        for w in workers:
            try:
                w['proc'].wait(timeout=60)
            except subprocess.TimeoutExpired:
                if owned(w):
                    os.killpg(w['proc'].pid, signal.SIGKILL)
                w['proc'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:
            u.write(run / f'RESOURCE_RELEASE_{time.time_ns()}.json', dict(result=placeholder('release')))
        u.write(run / 'sessions' / f'{time.time_ns()}.json', dict(status=state['status'], wall_seconds=time.time() - began))
        save(); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.run, args.resume)
