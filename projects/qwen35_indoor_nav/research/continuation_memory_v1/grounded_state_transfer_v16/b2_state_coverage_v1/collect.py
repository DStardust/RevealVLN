"""Bounded physical coverage repair from registered parent histories, then CPU review."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
V16 = HERE.parent
sys.path.insert(0, str(V16))
from v16_common import LINE, read, write, append, sha, digest, load, immutable
from continuation_service import NoInteriorJoin
from evaluator_v16 import legacy, state_sequence
transport = load('coverage_frozen_transport', V16/'identifiable_data_v1/collect_r2.py')
sys.path.insert(0, str(HERE))
admission = load('coverage_admission', HERE/'certify.py')
certify, HISTORIES, QUERIES = admission.certify, admission.HISTORIES, admission.QUERIES


def proposals(parent, config):
    for amount in config['turn_counts']:
        for direction in config['turn_directions']:
            away = [direction]*amount
            back = [('R' if direction == 'L' else 'L')]*amount
            yield dict(direction=direction, amount=amount,
                histories={h:parent['histories'][h]+away for h in HISTORIES},
                suffixes=dict(direct_stop=['S'], return_terminal=back+['S'],
                              acquire_anchor=back+parent['suffixes']['acquire_anchor']))


def source_prefix_matches(trace, reference, length):
    if trace['actions'][:length] != reference['actions'][:length]:
        return False
    for a,b in zip(trace['observations'][:length+1], reference['observations'][:length+1]):
        if any(a[key] != b[key] for key in ('rgb_hash','semantic_hash','pose')):
            return False
    return len(trace['observations']) >= length+1 and len(reference['observations']) >= length+1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); config = read(args.config)
    if not args.run_id.replace('_','').isalnum(): raise ValueError('RUN_ID')
    run = HERE/'runs'/args.run_id
    if run.exists() and not args.resume: raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (run/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = read(HERE/'SOURCE_LOCK.json')
    for name, expected in sources['files'].items():
        if sha(LINE/name) != expected: raise ValueError('SOURCE_CHANGED:'+name)
    immutable(run/'BINDING.json', dict(config=config, sources=sources))
    dataset_path = LINE/config['parent_dataset']
    if sha(dataset_path) != config['parent_dataset_sha256']: raise ValueError('PARENT_DATA_CHANGED')
    parents = {f['family_id']:f for f in read(dataset_path)['families']}
    selected = [parents[f] for f in config['parent_order']]
    if any(f['split'] not in ('FIT','DEV') for f in selected): raise ValueError('TEST_NOT_AUTHORIZED')
    began = time.monotonic(); last_check = float('-inf'); backend = None; complete = []; errors = []; attempts = 0
    previous = read(run/'STATUS.json').get('elapsed_seconds',0) if (run/'STATUS.json').exists() else 0
    store = transport.prior.ContentStore(run/'content', config['artifact_gib']*2**30)
    def check():
        nonlocal last_check
        elapsed = previous + time.monotonic()-began
        if elapsed > config['max_seconds']: raise TimeoutError('COLLECTION_TIME_LIMIT')
        if elapsed-last_check < 10: return
        last_check = elapsed
        p = subprocess.run(['nvidia-smi','-i',str(config['gpu']),
            '--query-gpu=uuid,memory.free,memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=15)
        fields = p.stdout.strip().split(',')
        if fields[0].strip() != config['gpu_uuid'] or int(fields[1]) < config['min_free_mib']:
            raise RuntimeError('GPU_IDENTITY_OR_HEADROOM')
        rss = int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
        if rss > config['rss_gib']*2**30: raise RuntimeError('RSS_LIMIT')
        append(run/'RESOURCES.jsonl',dict(elapsed=elapsed,gpu=p.stdout.strip(),rss=rss,pid=os.getpid()))
    def status(state, **fields):
        write(run/'STATUS.json',dict(status=state,certified_counterparts=len(complete),planned_counterparts=len(selected),
            rejected_parents=len(errors),attempts_this_session=attempts,elapsed_seconds=previous+time.monotonic()-began,
            new_training_updates=0,base_model_loaded=False,content_bytes=store.bytes,**fields))
    try:
        check()
        immutable(run/'RUNTIME_IDENTITY.json',dict(gpu=config['gpu'],gpu_uuid=config['gpu_uuid'],
            python=sys.executable,pid=os.getpid(),uid=os.getuid(),gid=os.getgid(),
            base_model_loaded=False,checkpoint_updates=0,sensors='Inherited NoInteriorJoin, RGB224, original SEE2; no STOP observation'))
        for rank,parent in enumerate(selected):
            folder = run/parent['family_id']; folder.mkdir(exist_ok=True)
            if (folder/'FAMILY.json').exists():
                complete.append(read(folder/'FAMILY.json')); continue
            if (folder/'EXHAUSTED.json').exists():
                errors.append(parent['family_id']); continue
            references = {}
            for h in HISTORIES:
                ref = parent['traces'][h+'__direct_stop']; path = LINE/ref['path']
                if sha(path) != ref['sha256']: raise ValueError('PARENT_TRACE_CHANGED')
                references[h] = read(path)
            backend = NoInteriorJoin(parent['scene'],config['gpu'],parent['roles'],store,
                dict(runtime_allowed=True,scene_glb=parent['scene'],gpu_device=config['gpu']))
            compiler = legacy.Compiler(**parent['compiler'])
            plans = list(proposals(parent,config)); immutable(folder/'PROPOSALS.json',plans)
            for index,plan in enumerate(plans):
                check(); trial = folder/f'attempt_{index:02d}'
                if (trial/'REJECTED.json').exists(): continue
                if trial.exists():
                    # A partial physical family is not resumed as a completed pair.
                    old = trial; trial = folder/f'attempt_{index:02d}_resume_{time.time_ns()}'
                    append(run/'ATTEMPTS.jsonl',dict(stage='INCOMPLETE_ATTEMPT_RETAINED',path=str(old.relative_to(LINE))))
                trial.mkdir(exist_ok=False); attempts += 1
                append(run/'ATTEMPTS.jsonl',dict(parent=parent['family_id'],stage='PROPOSAL_START',proposal=index,path=str(trial.relative_to(LINE))))
                cutoff = len(plan['histories']['seen']); old_cutoff = len(parent['histories']['seen'])
                traces = {}
                # First physical trace screens evidence and geometry, never policy performance.
                probe = transport.run_trace(backend,parent['initial_position'],plan['histories']['missing']+['S'],check)
                write(trial/'probe.json',probe,True)
                if not source_prefix_matches(probe,references['missing'],old_cutoff):
                    raise ValueError('PARENT_PREFIX_TRANSPORT_CHANGED')
                z = state_sequence(compiler,probe['observations'],'task_A')[-1]
                if probe['collisions'] or z[1] or z[2]:
                    reason = 'MISSING_EVENT_OR_TERMINAL_COVERAGE_NOT_MATCHED'
                    write(trial/'REJECTED.json',dict(reason=reason,state=z),True)
                    append(run/'ATTEMPTS.jsonl',dict(parent=parent['family_id'],stage='REJECTED',proposal=index,reason=reason)); continue
                for h,q in itertools.product(HISTORIES,QUERIES):
                    trace = probe if (h,q)==('missing','direct_stop') else transport.run_trace(
                        backend,parent['initial_position'],plan['histories'][h]+plan['suffixes'][q],check)
                    if not source_prefix_matches(trace,references[h],old_cutoff):
                        raise ValueError('PARENT_PREFIX_TRANSPORT_CHANGED')
                    traces[h+'__'+q] = trace
                    write(trial/(h+'__'+q+'.json'),trace,True)
                try:
                    certificate = certify(compiler,plan['histories'],plan['suffixes'],traces)
                except ValueError as exc:
                    # Expected physical proposal rejection is retained in the full funnel.
                    write(trial/'REJECTED.json',dict(reason=str(exc)),True)
                    append(run/'ATTEMPTS.jsonl',dict(parent=parent['family_id'],stage='REJECTED',proposal=index,reason=str(exc))); continue
                family = dict(parent, family_id='ABSENT_'+digest([parent['family_id'],plan])[:20],
                    parent_family_id=parent['family_id'],histories=plan['histories'],suffixes=plan['suffixes'],
                    certificate=certificate,terminal_present_at_takeover=False,
                    traces={k:dict(path=str((trial/(k+'.json')).relative_to(LINE)),sha256=sha(trial/(k+'.json'))) for k in traces},
                    content_root=str(store.root.relative_to(LINE)),
                    training_admission='CONTROLLED_MECHANISM_ONLY_PENDING_DATASET_AUDIT',
                    scope='New terminal-absent causal extensions of existing parent families; stationary reorientation, not translational recovery')
                write(folder/'FAMILY.json',family,True); complete.append(family)
                append(run/'ATTEMPTS.jsonl',dict(parent=parent['family_id'],stage='CERTIFIED',proposal=index,family_id=family['family_id'])); break
            if not (folder/'FAMILY.json').exists():
                write(folder/'EXHAUSTED.json',dict(proposals=len(plans)),True); errors.append(parent['family_id'])
            backend.close(); backend = None
            status('COLLECTING',parent=parent['family_id'],parents_processed=rank+1)
            print(f"parents={rank+1}/{len(selected)} counterparts={len(complete)} rejected_parents={len(errors)}",flush=True)
        immutable(run/'DATASET.json',dict(families=complete,all_parent_ids=config['parent_order'],
            exhausted_parent_ids=errors,formal_TEST_accessed=False,new_independent_houses=0,
            parent_dataset=config['parent_dataset'],all_training_arms_share_pool=True))
        status('CPU_REVIEW')
        verification = run/f'verification_{len(list(run.glob("verification_*"))):03d}'
        subprocess.run([sys.executable,'-I','-B',str(HERE/'review.py'),'--run',str(run),'--output',str(verification)],check=True)
        status('COMPLETE' if len(complete)==len(selected) else 'COMPLETE_WITH_SHORTFALL',
               result=str((verification/'RESULT.json').relative_to(LINE)))
    except BaseException as exc:
        write(run/f'FAILURE_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        status('STOPPED',error=repr(exc)); raise
    finally:
        if backend: backend.close()


if __name__ == '__main__': main()
