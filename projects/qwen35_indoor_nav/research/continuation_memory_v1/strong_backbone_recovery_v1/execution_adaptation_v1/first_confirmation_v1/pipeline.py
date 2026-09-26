"""Independent six-GPU training, then complete-group unseen evaluation."""
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

REVISION = Path(__file__).resolve().parent
HERE = REVISION
ADAPT = HERE.parent
BASE = ADAPT.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u
from transfer_pipeline import placeholder
from unseen_pipeline import devices
from review import records, summary, review


def process_start(pid):
    value = Path(f'/proc/{pid}/stat').read_text()
    return value[value.rfind(')') + 2:].split()[19]


def owned(worker):
    if worker['proc'].poll() is not None:
        return False
    try:
        return os.getpgid(worker['proc'].pid) == worker['proc'].pid and process_start(worker['proc'].pid) == worker['ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def completed_models(train, protocol, verify=True):
    result = []
    for arm in protocol['models']:
        path = train / arm / 'RESULT.json'
        if path.exists() and u.read(path)['status'] == 'COMPLETE':
            row = u.read(path)
            if row['completed_steps'] != protocol['training_steps'] or (verify and u.sha(train / arm / 'FINAL.pt') != row['final_sha256']):
                raise ValueError('TRAINED_MODEL_CHANGED')
            result.append(arm)
    return result


def unfinished_attempts(run):
    """Reject survivors before resume; charge interrupted attempts conservatively."""
    for start in (run / 'attempts').glob('*.start.json'):
        finish = start.with_name(start.name.replace('.start.json', '.json'))
        if finish.exists():
            continue
        row = u.read(start)
        try:
            if process_start(row['pid']) == row['ticks']:
                raise RuntimeError('PREVIOUS_WORKER_STILL_RUNNING')
        except (FileNotFoundError, ProcessLookupError):
            pass
        u.write(finish, dict(row, returncode=None, wall_seconds=max(0., time.time() - row['started_unix']),
            timing_scope='CONSERVATIVE_UPPER_BOUND_AFTER_INTERRUPTION'))


def main(run, resume):
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    u.verify_sources(run)
    p = u.read(run / 'PROTOCOL.json'); train = Path(p['training_run']); unseen = run / 'unseen'
    if (run / 'RESULT.json').exists():
        raise ValueError('CLOSED_EXPERIMENT')
    if (run / 'STARTED.json').exists() and not resume:
        raise ValueError('RESUME_REQUIRED')
    if not (run / 'STARTED.json').exists():
        u.write(run / 'STARTED.json', dict(unix=time.time(), pid=os.getpid()))
    for name in ('attempts', 'sessions', 'failed_attempts'):
        (run / name).mkdir(exist_ok=True)
    unfinished_attempts(run)
    for session in (unseen / 'evaluation').glob('*'):
        if not (session / 'STATE_SEAL.json').exists():
            session.rename(run / 'failed_attempts' / (session.name + '_' + str(time.time_ns())))
    hours = dict(p['inherited_phase_gpu_hours'])
    for path in (run / 'attempts').glob('*.json'):
        if not path.name.endswith('.start.json'):
            receipt = u.read(path)
            if receipt['gpu'] is not None:
                hours[receipt['cost_phase']] += receipt['wall_seconds'] / 3600
    wall_before = p['inherited_wall_seconds'] + sum(u.read(path)['wall_seconds'] for path in (run / 'sessions').glob('*.json'))
    began = time.time(); workers = []; leased = False
    state = dict(status='RUNNING', phase='PREPARE_FROZEN_HEADS', frozen_models=len(p['models']), planned_optimizer_steps=0,
        planned_groups=p['planned_groups'], planned_executions=p['planned_executions'])

    def save():
        costs = dict(hours)
        for w in workers:
            if w['gpu'] is not None:
                costs[w['cost_phase']] += (time.time() - w['start']) / 3600
        groups = records(unseen)
        state.update(unix=time.time(), phase_gpu_hours=costs, gpu_hours=sum(costs.values()),
            complete_models=completed_models(train, p, verify=False), sealed_groups=len(groups),
            recorded_groups=len(list((unseen / 'evaluation').glob('*/episodes/*/COMPLETE.json'))),
            models={arm:u.read(train / arm / 'PROGRESS.json') if (train / arm / 'PROGRESS.json').exists() else None for arm in p['models']},
            workers=[dict(pid=w['proc'].pid, gpu=w['gpu'], uuid=w['uuid'], output=str(w['out']),
                progress=u.read(w['out'] / 'PROGRESS.json') if (w['out'] / 'PROGRESS.json').exists() else None) for w in workers])
        u.write(run / 'STATUS.json', state)

    def spawn(command, gpu, output, cost_phase):
        inventory = devices() if gpu is not None else {}
        if gpu is not None and inventory[gpu]['free_mib'] < p['minimum_free_mib']:
            raise RuntimeError('INSUFFICIENT_FREE_GPU_MEMORY')
        uuid = inventory[gpu]['uuid'] if gpu is not None else ''
        path = run / 'attempts' / f'{state["phase"]}_{gpu}_{time.time_ns()}.log'
        log = path.open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        proc = subprocess.Popen(command, cwd=u.REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        w = dict(proc=proc, ticks=process_start(proc.pid), start=time.time(), gpu=gpu, uuid=uuid,
            out=output, path=path, log=log, command=command, cost_phase=cost_phase)
        workers.append(w)
        u.write(path.with_suffix('.start.json'), dict(pid=proc.pid, ticks=w['ticks'], started_unix=w['start'],
            gpu=gpu, uuid=uuid, output=str(output), command=command, cost_phase=cost_phase))

    def finish(w):
        elapsed = time.time() - w['start']
        if w['gpu'] is not None:
            hours[w['cost_phase']] += elapsed / 3600
        w['log'].close()
        u.write(w['path'].with_suffix('.json'), dict(pid=w['proc'].pid, ticks=w['ticks'], gpu=w['gpu'],
            uuid=w['uuid'], returncode=w['proc'].returncode, wall_seconds=elapsed,
            output=str(w['out']), command=w['command'], cost_phase=w['cost_phase']))

    def wait(refill=None):
        last_resource = 0
        while workers:
            for w in list(workers):
                if w['proc'].poll() is not None:
                    finish(w); workers.remove(w)
                    if w['proc'].returncode:
                        raise RuntimeError('WORKER_FAILED:' + str(w['path']))
                    if refill:
                        refill(w['gpu'])
            save()
            if any(state['phase_gpu_hours'][key] > p['phase_gpu_hour_limits'][key] for key in hours):
                raise RuntimeError('GPU_SESSION_BUDGET_REACHED')
            if wall_before + time.time() - began > p['wall_hours_limit'] * 3600:
                raise RuntimeError('WALL_BUDGET_REACHED')
            if time.time() - last_resource > 30:
                inventory = devices()
                rss = 0
                for w in workers:
                    if owned(w):
                        status = Path(f'/proc/{w["proc"].pid}/status').read_text().splitlines()
                        rss += next(int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:'))
                        if w['gpu'] is not None and inventory[w['gpu']]['free_mib'] < 512:
                            raise RuntimeError('GPU_HEADROOM_EXHAUSTED')
                state['worker_rss_bytes'] = rss
                state['gpu_inventory'] = inventory
                if rss > p['rss_limit_gib'] * 1024**3:
                    raise RuntimeError('WORKER_RSS_LIMIT')
                state['artifact_bytes'] = sum(path.stat().st_size for root in (train, run) for path in root.rglob('*') if path.is_file())
                if state['artifact_bytes'] > p['artifact_limit_gib'] * 1024**3:
                    raise RuntimeError('ARTIFACT_LIMIT')
                if (unseen / 'PROTOCOL.json').exists():
                    u.write(unseen / 'LIVE_RESULT.json', summary(unseen))
                last_resource = time.time()
            if workers:
                time.sleep(5)

    def interrupted(signum, frame):
        raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        save()
        before = placeholder('status')
        if before['leases'] or before['manual_paused'] or before['external_pids']:
            raise RuntimeError('RESOURCE_ALREADY_IN_USE')
        acquired = placeholder('acquire'); leased = True
        u.write(run / f'RESOURCE_LEASE_{time.time_ns()}.json', dict(before=before, acquired=acquired))
        inventory = devices()
        available = [g for g in p['gpu_indices'] if inventory[g]['free_mib'] >= p['minimum_free_mib']]
        if not available:
            raise RuntimeError('NO_USABLE_REGISTERED_DEVICE')
        if set(completed_models(train, p)) != set(p['models']):
            raise ValueError('FROZEN_HEAD_MISSING_NO_RETRAIN_ALLOWED')
        if not (unseen / 'EXECUTOR_BINDING.json').exists():
            raise ValueError('EVALUATION_MUST_BE_PREREGISTERED')
        binding = u.read(unseen / 'EXECUTOR_BINDING.json')
        if binding['sha256'] != u.sha(REVISION / 'evaluate_worker.py'):
            raise ValueError('EXECUTOR_REVISION_CHANGED')
        u.verify_sources(unseen)
        manifest = u.read(unseen / 'DATA_MANIFEST.json')['episodes']
        if len(manifest) != p['planned_groups']:
            raise ValueError('UNSEEN_DENOMINATOR_CHANGED')
        done = records(unseen)
        smoke_id = p['first_complete_group_id']
        def evaluate(gpu, ids):
            output = unseen / 'evaluation' / f'gpu{gpu}_{time.time_ns()}'
            spawn([str(u.PYTHON), '-I', '-B', str(REVISION / 'evaluate_worker.py'), '--run', str(unseen),
                '--output', str(output), '--ids', ','.join(map(str, ids))], gpu, output, 'EVALUATE')
        if smoke_id not in done:
            state['phase'] = 'FIRST_UNSEEN_GROUP'; save()
            evaluate(available[0], [smoke_id]); wait()
        if smoke_id not in records(unseen):
            raise ValueError('FIRST_UNSEEN_GROUP_NOT_SEALED')
        u.write(run / 'FIRST_GROUP_RESULT.json', dict(status='REAL_SEVEN_ARM_GROUP_SEALED', id=smoke_id,
            included_in_full_denominator=True, score_not_used_for_continuation=True))
        state['phase'] = 'UNSEEN'; save()
        done = records(unseen)
        pending = deque(e['id'] for e in manifest if e['id'] not in done)
        def eval_next(gpu):
            if pending:
                evaluate(gpu, [pending.popleft() for _ in range(min(p['chunk_groups'], len(pending)))])
        for gpu in available:
            eval_next(gpu)
        wait(eval_next)
        state['phase'] = 'REVIEW'; save()
        result = review(unseen)
        u.write(run / 'RESULT.json', dict(status='COMPLETE', frozen_models_reused=len(p['models']),
            planned_optimizer_steps=0,
            phase_gpu_hours=hours, gpu_hours=sum(hours.values()), base_updates=0,
            evaluation_result=str(unseen / 'RESULT.json'), old_candidate_retained=True,
            complete_groups=result['complete_groups']))
        state.update(status='COMPLETE', phase='REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, InterruptedError) else 'FAILED', error=repr(error))
        u.write(run / f'FAILURE_{time.time_ns()}.json', dict(state, traceback=traceback.format_exc()))
        raise
    finally:
        for w in workers:
            if owned(w):
                os.killpg(w['proc'].pid, signal.SIGTERM)
        for w in workers:
            try:
                w['proc'].wait(timeout=45)
            except subprocess.TimeoutExpired:
                if owned(w):
                    os.killpg(w['proc'].pid, signal.SIGKILL)
                w['proc'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:
            u.write(run / f'RESOURCE_RELEASE_{time.time_ns()}.json', dict(released=placeholder('release')))
        u.write(run / 'sessions' / f'{time.time_ns()}.json', dict(status=state['status'], wall_seconds=time.time()-began))
        save(); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.run.resolve(), args.resume)
