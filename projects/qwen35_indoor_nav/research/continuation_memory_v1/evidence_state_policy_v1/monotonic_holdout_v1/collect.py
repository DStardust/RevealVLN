"""Score-blind real house replay. No policy is loaded by this process."""
import itertools
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from continuation_service import ContentStore,NoInteriorJoin
from evaluator_v16 import legacy,state_sequence
transport=load('holdout_transport',V16/'identifiable_data_v1/collect_r2.py')
present=load('holdout_present_certificate',V16/'identifiable_data_v1/certify.py')
absent=load('holdout_absent_certificate',V16/'b2_state_coverage_v1/certify.py')
coverage=load('holdout_coverage_proposals',V16/'b2_state_coverage_v1/collect.py')

def whole_family(backend,position,plan,trial,compiler,checker,check,reference=None):
    traces={}
    for h,q in itertools.product(present.HISTORIES,plan['suffixes']):
        trace=transport.run_trace(backend,position,plan['histories'][h]+plan['suffixes'][q],check)
        if reference is not None and not coverage.source_prefix_matches(trace,reference[h],len(reference[h]['actions'])-1):
            raise ValueError('PARENT_PREFIX_TRANSPORT_CHANGED')
        traces[h+'__'+q]=trace;write(trial/(h+'__'+q+'.json'),trace,True)
    return checker(compiler,plan['histories'],plan['suffixes'],traces),traces

def main(run):
    cfg=runtime_config(run);verify_lock(read(run/'SOURCE_LOCK.json'))
    house_id=read(Path(os.environ['B2_DEVICE']))['house']
    house=next(h for h in read(HERE/'HOUSE_MANIFEST.json')['houses'] if h['house']==house_id)
    out=run/'collect'/house_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'DATASET.json').exists():return
    store=ContentStore(out/'content',cfg['artifact_gib']*2**30);began=time.monotonic()
    def check():
        if time.monotonic()-began>cfg['max_session_hours']*3600-120:raise TimeoutError('COLLECTION_SESSION_LIMIT')
    def log(**fields):append(out/'ATTEMPTS.jsonl',dict(house=house_id,unix=time.time(),**fields))
    def progress(stage):
        write(out/'STATUS.json',dict(stage=stage,parents=len(list(out.glob('position_*/FAMILY.json'))),
              variants=len(list(out.glob('position_*/FAMILY.json')))+len(list(out.glob('absent_*/FAMILY.json'))),
              target_parents=cfg['families_per_house'],seconds=time.monotonic()-began,base_model_loaded=False))
    def trial_dir(folder,name):
        trial=folder/name
        if trial.exists():
            log(stage='PARTIAL_ATTEMPT_RETAINED',path=str(trial.relative_to(LINE)))
            trial=folder/(name+'_resume_'+str(time.time_ns()))
        trial.mkdir();return trial
    for path,expected in house['asset_sha256'].items():
        if sha(Path(path))!=expected:raise ValueError('SCENE_ASSET_CHANGED')
    roles={'r'+str(i):r['spec'] for i,r in enumerate(house['roles']) if not set(r['eligible'])&{0,65535,4294967295}}
    planner=load('holdout_reserved_metadata',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    adapter=load('holdout_role_match',LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py')
    objects=planner.parse_house(Path(house['scene']).with_suffix('.house').read_text())
    reserved=[o for i,o in objects.items() if i in adapter.RESERVED_MASKS]
    excluded={k:v for k,v in roles.items() if any(adapter.role_matches(o,v) for o in reserved)}
    immutable(out/'RESERVED_EXCLUSIONS.json',excluded);roles={k:v for k,v in roles.items() if k not in excluded}
    groups=transport.catalog(V16/'identifiable_data_v1/balanced_probe_001')
    parents=[read(p) for p in sorted(out.glob('position_*/FAMILY.json'))];backend=None
    try:
        backend=NoInteriorJoin(house['scene'],cfg['gpu'],roles,store,dict(runtime_allowed=True,scene_glb=house['scene'],gpu_device=cfg['gpu']))
        for key,ids in backend.eligible.items():
            if ids!=house['roles'][int(key[1:])]['eligible']:raise ValueError('SEMANTIC_INVENTORY_CHANGED')
        if not (out/'POSITIONS.json').exists():immutable(out/'POSITIONS.json',transport.positions(backend,cfg['positions_per_house'],cfg['seed']))
        for index,position in enumerate(read(out/'POSITIONS.json')):
            if len(parents)>=cfg['families_per_house']:break
            check();folder=out/f'position_{index:03d}';folder.mkdir(exist_ok=True)
            if (folder/'DONE.json').exists() or (folder/'FAMILY.json').exists():continue
            if not (folder/'PANORAMA.json').exists():
                immutable(folder/'PANORAMA.json',transport.run_trace(backend,position,['L']*24,check))
            delay=cfg['delay_actions'][index%len(cfg['delay_actions'])]
            plans=transport.plans(read(folder/'PANORAMA.json'),groups,backend.eligible,delay)[:cfg['plans_per_position']]
            immutable(folder/'PROPOSALS.json',plans);log(stage='DISCOVERY',position=index,proposals=len(plans))
            for j,plan in enumerate(plans):
                if (folder/f'proposal_{j:02d}'/'REJECTED.json').exists():continue
                trial=trial_dir(folder,f'proposal_{j:02d}')
                roles2={key:roles[plan[key]] for key in ('anchor','terminal')}
                a,t=('the '+roles2[k]['raw_match']['value']+' in the '+roles2[k]['room'] for k in ('anchor','terminal'))
                instruction=f'Before stopping at {t}, establish a two-observation sighting of {a}. Stop only on a subsequent two-observation sighting of {t}.'
                spec=dict(roles={k:[v['mpcat40'],v['room']] for k,v in roles2.items()},
                    tasks=dict(task_A=dict(anchor='anchor',terminal='terminal',instruction=instruction)),
                    eligible={k:backend.eligible[plan[k]] for k in roles2})
                log(stage='REPLAY_START',position=index,proposal=j)
                try:certificate,traces=whole_family(backend,position,plan,trial,legacy.Compiler(**spec),present.certify,check)
                except ValueError as exc:
                    immutable(trial/'REJECTED.json',dict(reason=str(exc)));log(stage='REJECTED',position=index,proposal=j,reason=str(exc));continue
                family_id='HOLDOUT_'+digest([house_id,position,plan])[:20]
                family=dict(family_id=family_id,parent_family_id=family_id,parent_slot=len(parents),house=house_id,split='TEST',
                    stratum='terminal_present',scene=house['scene'],initial_position=position,initial_yaw=0,compiler=spec,roles=roles2,
                    terminal_instruction=f'Stop at {t} only once it has been seen in two consecutive observations.',
                    histories=plan['histories'],suffixes=plan['suffixes'],certificate=certificate,
                    traces={k:dict(path=str((trial/(k+'.json')).relative_to(LINE)),sha256=sha(trial/(k+'.json'))) for k in traces},
                    content_root=str(store.root.relative_to(LINE)),training_admission=False,
                    evaluation_scope='Frozen-head stationary SEE2 holdout, never training data in this run')
                immutable(folder/'FAMILY.json',family);parents.append(family);log(stage='CERTIFIED_PRESENT',family=family_id);break
            immutable(folder/'DONE.json',dict(accepted=(folder/'FAMILY.json').exists()));progress('COLLECT_PRESENT')
        backend.close();backend=None
        for parent in parents:
            folder=out/f"absent_{parent['parent_slot']:02d}";folder.mkdir(exist_ok=True)
            if (folder/'FAMILY.json').exists() or (folder/'EXHAUSTED.json').exists():continue
            references={h:read(LINE/parent['traces'][h+'__direct_stop']['path']) for h in present.HISTORIES}
            backend=NoInteriorJoin(parent['scene'],cfg['gpu'],parent['roles'],store,dict(runtime_allowed=True,scene_glb=parent['scene'],gpu_device=cfg['gpu']))
            compiler=legacy.Compiler(**parent['compiler']);plans=list(coverage.proposals(parent,cfg));immutable(folder/'PROPOSALS.json',plans)
            for j,plan in enumerate(plans):
                check()
                if (folder/f'proposal_{j:02d}'/'REJECTED.json').exists():continue
                trial=trial_dir(folder,f'proposal_{j:02d}');log(stage='ABSENT_START',parent=parent['family_id'],proposal=j)
                probe=transport.run_trace(backend,parent['initial_position'],plan['histories']['missing']+['S'],check)
                immutable(trial/'PROBE.json',probe)
                if not coverage.source_prefix_matches(probe,references['missing'],len(parent['histories']['missing'])):raise ValueError('PARENT_PREFIX_TRANSPORT_CHANGED')
                z=state_sequence(compiler,probe['observations'],'task_A')[-1]
                if probe['collisions'] or z[1] or z[2]:
                    immutable(trial/'REJECTED.json',dict(reason='MISSING_OR_TERMINAL_NOT_ABSENT',state=z));log(stage='REJECTED_ABSENT',proposal=j,parent=parent['family_id']);continue
                try:certificate,traces=whole_family(backend,parent['initial_position'],plan,trial,compiler,absent.certify,check,references)
                except ValueError as exc:
                    if 'TRANSPORT' in str(exc):raise
                    immutable(trial/'REJECTED.json',dict(reason=str(exc)));log(stage='REJECTED_ABSENT',proposal=j,reason=str(exc));continue
                family=dict(parent,family_id='ABSENT_'+digest([parent['family_id'],plan])[:20],stratum='terminal_absent',
                    histories=plan['histories'],suffixes=plan['suffixes'],certificate=certificate,
                    traces={k:dict(path=str((trial/(k+'.json')).relative_to(LINE)),sha256=sha(trial/(k+'.json'))) for k in traces})
                immutable(folder/'FAMILY.json',family);log(stage='CERTIFIED_ABSENT',family=family['family_id']);break
            if not (folder/'FAMILY.json').exists():immutable(folder/'EXHAUSTED.json',dict(proposals=len(plans)))
            backend.close();backend=None;progress('COLLECT_ABSENT')
        families=parents+[read(p) for p in sorted(out.glob('absent_*/FAMILY.json'))]
        immutable(out/'DATASET.json',dict(families=families,planned=cfg['families_per_house']*2,policy_scores_used=False))
        progress('COLLECTION_COMPLETE' if len(families)==cfg['families_per_house']*2 else 'PHYSICAL_SHORTFALL')
    finally:
        if backend:backend.close()

if __name__=='__main__':main(Path(sys.argv[1]))
