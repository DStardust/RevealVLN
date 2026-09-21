"""Generate matched physical histories with unchanged actions/sensors/SEE2 semantics."""
import argparse
from collections import defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from continuation_service import NoInteriorJoin
from v16_common import read, write, append, sha, digest, load, LINE
from evaluator_v16 import legacy
# Avoid colliding with this module's filename.
prior = load('ident_frozen_collector', HERE.parent/'collect.py')
sys.path.insert(0, str(HERE))
from certify import certify, HISTORIES, QUERIES


def catalog(probe):
    result = read(probe/'RESULT.json')
    groups = []
    for ids in result['matched_groups'].values():
        if len(ids) < 4: continue
        groups.append([read(probe/f'H_{i:04d}.json')['actions'] for i in ids])
    return groups


def witness_roles(a, b, eligible):
    return {role for role, ids in eligible.items()
            if any(a.get(str(i),0)>=256 and b.get(str(i),0)>=256 for i in ids)}


def plans(panorama, groups, eligible, delay):
    pixels = [o['pixels'] for o in panorama['observations'][:24]]
    current = witness_roles(pixels[1],pixels[0],eligible)
    if not current:return []
    options = []
    for group_id, words in enumerate(groups):
        candidates = []
        for word in words:
            yaw=0;seen=set();recent=set()
            for t,action in enumerate(word):
                next_yaw=(yaw+(1 if action=='L' else -1))%24
                event=witness_roles(pixels[yaw],pixels[next_yaw],eligible)
                seen |= event
                if t>=len(word)-8:recent |= event
                yaw=next_yaw
            candidates.append((word,seen,recent))
        for anchor in sorted(set().union(*(v[1] for v in candidates))):
            yes=[w for w,s,r in candidates if anchor in s and anchor not in r]
            no=[w for w,s,r in candidates if anchor not in s]
            if len(yes)<2 or len(no)<2:continue
            for terminal in sorted(current):
                if terminal==anchor or set(eligible[terminal]) & set(eligible[anchor]):continue
                wrong=[]
                for direction in ('L','R'):
                    yaw=0
                    for count in range(1,13):
                        nxt=(yaw+(1 if direction=='L' else -1))%24
                        if count>=2 and terminal not in witness_roles(pixels[yaw],pixels[nxt],eligible):
                            wrong=[direction]*count+['S'];break
                        yaw=nxt
                    if wrong:break
                if not wrong:continue
                histories={h:list(w)+['L','R']*((delay-8)//2)
                           for h,w in zip(HISTORIES,(yes[0],yes[1],no[0],no[1]))}
                options.append(dict(anchor=anchor,terminal=terminal,group=group_id,histories=histories,
                    suffixes=dict(direct_stop=['S'],acquire_anchor=list(yes[0])+['S'],wrong_terminal=wrong)))
    return options


def run_trace(backend, position, actions, check):
    backend.reset(position,0,0)
    trace=dict(actions=[],observations=[dict(backend.observe(),step=0)],collisions=0,complete=False,
               interior_state_assignments=0)
    for action in actions:
        check();trace['actions'].append(action)
        if action!='S':
            trace['collisions']+=int(backend.step(action))
            trace['observations'].append(dict(backend.observe(),step=len(trace['actions'])))
        if not trace['observations'][-1]['evidence_complete']:raise ValueError('UNKNOWN_SEMANTIC_EVIDENCE')
    if backend.counts['explicit_reconstructions']:raise ValueError('INTERIOR_STATE_ASSIGNMENT')
    trace['complete']=True
    return trace


def positions(backend, count, seed):
    rng=random.Random(seed)
    ids=sorted(set(i for values in backend.eligible.values() for i in values));rng.shuffle(ids)
    result=[]
    for i in ids:
        center=backend.objects[i]['center']
        for dx,dz in ((0,1.25),(.75,1.25),(-.75,1.25),(0,2.0)):
            raw=backend.np.asarray([center[0]+dx,center[1],center[2]+dz],dtype=backend.np.float32)
            point=backend.sim.pathfinder.snap_point(raw).tolist()
            if not all(math.isfinite(v) for v in point):continue
            if math.hypot(point[0]-raw[0],point[2]-raw[2])>.75:continue
            if any(math.dist(point,p)<1.0 for p in result):continue
            result.append(point)
            if len(result)==count:return result
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config);run=HERE/'runs'/args.run_id
    if not args.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    if run.exists() and not args.resume:raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source={str(p.relative_to(LINE)):sha(p) for p in (HERE/'collect.py',HERE/'certify.py',
            HERE.parent/'collect.py',HERE.parent/'evaluator_v16.py',HERE.parent/'continuation_service.py',
            LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')}
    binding=dict(config=cfg,sources=source)
    if (run/'BINDING.json').exists():
        if read(run/'BINDING.json')!=binding:raise ValueError('RESUME_BINDING_CHANGED')
    else:write(run/'BINDING.json',binding,True)
    began=time.monotonic();last=0;completed=[];backend=None
    def check():
        nonlocal last
        elapsed=time.monotonic()-began
        if elapsed>cfg['max_seconds']:raise TimeoutError('COLLECTION_TIME_LIMIT')
        if elapsed-last<15:return
        last=elapsed
        result=subprocess.run(['nvidia-smi','-i',str(cfg['gpu']),'--query-gpu=uuid,memory.free,memory.used',
                               '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20,check=True)
        fields=result.stdout.strip().split(',');free=int(fields[1])
        if fields[0].strip()!=cfg['gpu_uuid'] or free<cfg['min_free_mib']:raise RuntimeError('GPU_IDENTITY_OR_HEADROOM')
        rss=int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
        if rss>cfg['rss_gib']*2**30:raise RuntimeError('RSS_LIMIT')
        append(run/'RESOURCES.jsonl',dict(elapsed=elapsed,gpu=result.stdout.strip(),rss=rss,pid=os.getpid()))
    manifest=read(HERE.parent/'DATA_MANIFEST.json');groups=catalog(HERE/'balanced_probe_001')
    store=prior.ContentStore(run/'content',cfg['artifact_gib']*2**30)
    status='FAILED';error=None
    try:
        check()
        for house in manifest['houses']:
            if house['split'] not in ('FIT','DEV'):continue
            house_dir=run/house['house'];house_dir.mkdir(exist_ok=True)
            accepted=[read(p) for p in sorted(house_dir.glob('*/FAMILY.json'))]
            completed.extend(accepted)
            if len(accepted)>=cfg['families_per_house']:continue
            # Remove whole signatures that contain a reserved mask before opening Habitat.
            roles={'r'+str(i):r['spec'] for i,r in enumerate(house['roles'])
                   if not set(r['eligible']) & {0,65535,4294967295}}
            backend=NoInteriorJoin(house['scene'],cfg['gpu'],roles,store,
                        dict(runtime_allowed=True,scene_glb=house['scene'],gpu_device=cfg['gpu']))
            for key,ids in backend.eligible.items():
                if ids!=house['roles'][int(key[1:])]['eligible']:raise ValueError('SEMANTIC_INVENTORY_CHANGED')
            posfile=house_dir/'POSITIONS.json'
            if not posfile.exists():write(posfile,positions(backend,cfg['positions_per_house'],cfg['seed']),True)
            for index,position in enumerate(read(posfile)):
                check()
                folder=house_dir/f'position_{index:03d}';folder.mkdir(exist_ok=True)
                if (folder/'DONE.json').exists():continue
                if (folder/'PANORAMA.json').exists():pano=read(folder/'PANORAMA.json')
                else:
                    pano=run_trace(backend,position,['L']*24,check)
                    write(folder/'PANORAMA.json',pano,True)
                delay=cfg['delay_actions'][index%len(cfg['delay_actions'])]
                candidates=plans(pano,groups,backend.eligible,delay)[:cfg['plans_per_position']]
                if (folder/'PROPOSALS.json').exists():
                    if read(folder/'PROPOSALS.json')!=candidates:raise ValueError('RESUME_PROPOSALS_CHANGED')
                else:write(folder/'PROPOSALS.json',candidates,True)
                append(run/'ATTEMPTS.jsonl',dict(house=house['house'],position=index,stage='DISCOVERY',proposals=len(candidates)))
                for proposal_index,plan in enumerate(candidates):
                    prior_trials=sorted(folder.glob(f'attempt_{proposal_index:02d}_*'))
                    if any((p/'REJECTED.json').exists() for p in prior_trials):continue
                    if (folder/'FAMILY.json').exists():break
                    trial=folder/f'attempt_{proposal_index:02d}_{len(prior_trials)+1:02d}';trial.mkdir(exist_ok=False)
                    roles2={name:roles[plan[name]] for name in ('anchor','terminal')}
                    anchor,terminal=roles2['anchor'],roles2['terminal']
                    a='the '+anchor['raw_match']['value']+' in the '+anchor['room']
                    t='the '+terminal['raw_match']['value']+' in the '+terminal['room']
                    instruction=(f'First see {a} in two consecutive observations, then see {t} in two consecutive observations, and stop immediately.' if house['split']=='FIT' else
                                 f'Before stopping at {t}, establish a two-observation sighting of {a}. Stop only on a subsequent two-observation sighting of {t}.')
                    terminal_instruction=(f'See {t} in two consecutive observations and stop immediately.' if house['split']=='FIT' else
                                          f'Stop at {t} only once it has been seen in two consecutive observations.')
                    spec=dict(roles={k:[v['mpcat40'],v['room']] for k,v in roles2.items()},
                              tasks=dict(task_A=dict(anchor='anchor',terminal='terminal',instruction=instruction)),
                              eligible={k:backend.eligible[plan[k]] for k in roles2})
                    compiler=legacy.Compiler(**spec);traces={}
                    append(run/'ATTEMPTS.jsonl',dict(house=house['house'],position=index,proposal=proposal_index,stage='REPLAY_START'))
                    try:
                        for h,q in itertools.product(HISTORIES,QUERIES):
                            trace=run_trace(backend,position,plan['histories'][h]+plan['suffixes'][q],check)
                            traces[h+'__'+q]=trace;write(trial/(h+'__'+q+'.json'),trace,True)
                        certificate=certify(compiler,plan['histories'],plan['suffixes'],traces)
                    except ValueError as exc:
                        write(trial/'REJECTED.json',dict(error=str(exc)),True)
                        append(run/'ATTEMPTS.jsonl',dict(house=house['house'],position=index,proposal=proposal_index,stage='REJECTED',reason=str(exc)))
                        continue
                    family_id='IDENT_'+digest([house['house'],position,plan])[:20]
                    family=dict(family_id=family_id,house=house['house'],split=house['split'],
                        scene=house['scene'],initial_position=position,initial_yaw=0,
                        compiler=spec,roles=roles2,terminal_instruction=terminal_instruction,
                        histories=plan['histories'],suffixes=plan['suffixes'],certificate=certificate,
                        traces={key:dict(path=str((trial/(key+'.json')).relative_to(LINE)),sha256=sha(trial/(key+'.json'))) for key in traces},
                        content_root=str(store.root.relative_to(LINE)),
                        training_admission='CONTROLLED_MECHANISM_ONLY_PENDING_DATASET_AUDIT',
                        scope='in_place_SEE2_history_conditioned_stopping; not translational recovery or natural navigation')
                    write(folder/'FAMILY.json',family,True);accepted.append(family);completed.append(family)
                    append(run/'ATTEMPTS.jsonl',dict(house=house['house'],position=index,proposal=proposal_index,stage='CERTIFIED',family_id=family_id))
                    break
                write(folder/'DONE.json',dict(accepted=(folder/'FAMILY.json').exists()),True)
                write(run/'STATUS.json',dict(status='COLLECTING',house=house['house'],positions=index+1,
                    house_accepted=len(accepted),total_accepted=len(completed),target=cfg['families_per_house']*5,
                    seconds=time.monotonic()-began,content_bytes=store.bytes))
                print(house['house'],index+1,len(accepted),len(completed),flush=True)
                if len(accepted)>=cfg['families_per_house']:break
            backend.close();backend=None
        status='COLLECTION_COMPLETE' if len(completed)==cfg['families_per_house']*5 else 'COLLECTION_SHORTFALL'
        write(run/'DATASET.json',dict(families=completed,scope='CONTROLLED_MECHANISM_DEVELOPMENT',
                    all_candidates_and_rejections='ATTEMPTS.jsonl',formal_TEST_accessed=False),True)
    except BaseException as exc:
        error=repr(exc);write(run/'FAILURE.json',dict(error=error,traceback=traceback.format_exc()),True)
        raise
    finally:
        if backend:backend.close()
        write(run/'STATUS.json',dict(status=status,accepted=len(completed),target=cfg['families_per_house']*5,
            error=error,seconds=time.monotonic()-began,content_bytes=store.bytes,
            completed_families=[f['family_id'] for f in completed]))

if __name__=='__main__':main()
