"""Bounded real history discovery and complete physical crossed executions."""
import itertools
import math
import os
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from evaluator_v16 import legacy, evaluate, state_sequence

class Rejected(RuntimeError):
    pass

class ContentStore:
    """Lossless content addressed arrays, never substituted by simulator metadata."""
    def __init__(self, root, max_bytes):
        self.root=Path(root); self.root.mkdir(exist_ok=True,parents=True)
        if self.root.is_symlink() or not self.root.resolve().is_relative_to(HERE):raise ValueError('CONTENT_PATH')
        self.bytes=sum(p.stat().st_size for p in self.root.iterdir());self.max_bytes=max_bytes;self.verified=set()
    def put_array(self, array, kind):
        import numpy as np
        h=hashlib.sha256(array.tobytes()).hexdigest();path=self.root/(h+'.'+kind+'.npy')
        if path.exists():
            if path not in self.verified:
                actual=np.load(path,allow_pickle=False)
                if actual.dtype!=array.dtype or actual.shape!=array.shape or not np.array_equal(actual,array):raise ValueError('CONTENT_CORRUPTION')
                self.verified.add(path)
        else:
            if self.bytes+array.nbytes+256>self.max_bytes:raise RuntimeError('ARTIFACT_LIMIT')
            tmp=path.with_suffix(path.suffix+'.tmp')
            with tmp.open('xb') as out:np.save(out,array,allow_pickle=False)
            os.link(tmp,path);tmp.unlink();self.bytes+=path.stat().st_size;self.verified.add(path)
        return h

class Runner:
    def __init__(self, backend, compiler, run, cap):
        self.backend=backend;self.compiler=compiler;self.run=run;self.cap=cap
    def start(self,position,yaw=0):
        self.backend.reset(position,yaw,0)
        self.trace=dict(actions=[],observations=[],collisions=0,complete=False,interior_state_assignments=0,
                        initial_position=position,initial_yaw=yaw,recovery=[])
        self.observe()
    def observe(self):
        o=dict(self.backend.observe(),step=len(self.trace['observations']))
        self.trace['observations'].append(o)
        if not o['evidence_complete']:raise Rejected('UNKNOWN_SEMANTIC_MASK')
        return o
    def move(self,action):
        if len(self.trace['actions'])>=self.cap:raise Rejected('DISCOVERY_BUDGET')
        self.trace['actions'].append(action)
        if action!='S':
            hit=self.backend.step(action);self.trace['collisions']+=int(hit);self.observe()
            if hit:raise Rejected('COLLISION')
    def navigate(self,target,center=None):
        b=self.backend
        follower=b.hs.GreedyGeodesicFollower(b.sim.pathfinder,b.sim.get_agent(0),goal_radius=.35)
        while True:
            position=self.trace['observations'][-1]['pose']['position']
            path=b.hs.ShortestPath();path.requested_start=b.np.asarray(position,dtype=b.np.float32);path.requested_end=b.np.asarray(target,dtype=b.np.float32)
            if not b.sim.pathfinder.find_path(path):raise Rejected('NO_PATH')
            if float(path.geodesic_distance)<=.35 and math.dist(position,target)<=.35:break
            before=digest(b._pose())
            try:a=follower.next_action_along(b.np.asarray(target,dtype=b.np.float32))
            except b.hs.errors.GreedyFollowerError:raise Rejected('FOLLOWER_ERROR')
            if digest(b._pose())!=before:raise ValueError('FOLLOWER_CHANGED_STATE')
            key={'move_forward':'F','turn_left':'L','turn_right':'R'}.get(a)
            if key is None:raise Rejected('FOLLOWER_EARLY_STOP')
            self.move(key)
        if center is not None:
            backend_module=load('v16_heading',LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py')
            pose=self.trace['observations'][-1]['pose'];q=pose['rotation']
            yaw=int(round(2*math.atan2(q[2],q[0])*12/math.pi))%24
            desired=backend_module.heading(center[0]-pose['position'][0],center[2]-pose['position'][2])
            for a in backend_module.turns(desired-yaw)+['L','R']:self.move(a)
    def witness(self,role,targets):
        for target in targets:
            self.navigate(target['target'],target['center'])
            # Predeclared active sweep; actual frames only, unchanged SEE2 threshold.
            for _ in range(24):
                if self.compiler.atoms(self.trace['observations'])[-1][role]:return
                self.move('L')
        raise Rejected('NO_VISIBLE_SEE2_WITNESS')
    def save(self,path):
        write(path,self.trace,True)

def instructions(roles,split):
    def name(role):
        r=roles[role];return 'the '+r['raw_match']['value']+' in the '+r['room']
    result={}
    for task,role in [('task_A','anchor_A'),('task_B','anchor_B')]:
        a,t=name(role),name('terminal')
        if split=='FIT':text=f'First see {a} in two consecutive observations, then see {t} in two consecutive observations, and stop immediately.'
        elif split=='DEV':text=f'Before stopping at {t}, establish a two-observation sighting of {a}. Stop only on a subsequent two-observation sighting of {t}.'
        else:text=f'Complete these sightings in order: {a}, followed by {t}. Each sighting requires two consecutive views. End by stopping on the final sighting.'
        result[task]=dict(anchor=role,terminal='terminal',instruction=text)
    return result

def terminal_instruction(roles,split):
    r=roles['terminal'];obj='the '+r['raw_match']['value']+' in the '+r['room']
    return {'FIT':f'See {obj} in two consecutive observations and stop immediately.',
            'DEV':f'Stop at {obj} only once it has been seen in two consecutive observations.',
            'TEST':f'Complete a sighting of {obj} across two consecutive views. End by stopping on that sighting.'}[split]

def collect_family(backend, proposal, house, run, cfg, store):
    folder=run/'proposals'/proposal['id'];folder.mkdir(parents=True,exist_ok=False)
    roles={name:house['roles'][idx]['spec'] for name,idx in zip(('anchor_A','anchor_B','terminal'),proposal['role_indices'])}
    eligible={name:backend.eligible['role_'+str(idx)] for name,idx in zip(roles,proposal['role_indices'])}
    compiler_config=dict(roles={k:[v['mpcat40'],v['room']] for k,v in roles.items()},tasks=instructions(roles,house['split']),eligible=eligible)
    compiler=legacy.Compiler(**compiler_config)
    targets=dict(zip(roles,proposal['targets']));histories={};recovery_records={};traces={}
    runner=Runner(backend,compiler,run,cfg['history_cap'])
    current_path=None
    try:
        selected_yaw=None
        for yaw in cfg['neutral_start_yaw_candidates']:
            current_path=folder/f'INITIAL_VIEW_PROBE_{yaw:02d}.json'
            runner.start(proposal['start'],yaw)
            for a in ['L','R']:runner.move(a)
            runner.trace['complete']=True;runner.save(current_path)
            events=compiler.atoms(runner.trace['observations'])
            if not any(e['anchor_A'] or e['anchor_B'] for e in events):selected_yaw=yaw;break
        if selected_yaw is None:raise Rejected('NO_ANCHOR_NEUTRAL_START_VIEW')
        for h,role,recovery in [('H_A','anchor_A',False),('H_B','anchor_B',False),('H_A_R','anchor_A',True),('H_B_R','anchor_B',True)]:
            current_path=folder/(h+'_DISCOVERY.json');runner.start(proposal['start'],selected_yaw)
            runner.witness(role,targets[role])
            if recovery:
                start=len(runner.trace['actions']);before=runner.trace['observations'][-1]['pose']['position']
                for action in cfg['perturbation']:runner.move(action)
                after=runner.trace['observations'][-1]['pose']['position']
                if math.dist(before,after)<.2:raise Rejected('PERTURBATION_NO_DISPLACEMENT')
                recovery_records[h]=dict(perturbation_start=start,actions=cfg['perturbation'],before=before,after=after,
                    meaning='wrong heading plus actual displaced step before resuming registered return target',teacher_recovery_end=None)
            angle=selected_yaw*math.pi/12
            runner.navigate(proposal['start'],[proposal['start'][0]-math.sin(angle),proposal['start'][1],proposal['start'][2]-math.cos(angle)])
            # A real public tail moves the anchor beyond the recent action window.
            for action in ['L','R']*5:runner.move(action)
            events=compiler.atoms(runner.trace['observations'])
            other='anchor_B' if role=='anchor_A' else 'anchor_A'
            if not any(e[role] for e in events) or any(e[other] for e in events):raise Rejected('HISTORY_EVENT_STATE_NOT_ISOLATED')
            runner.trace['complete']=True;runner.save(current_path)
            histories[h]=list(runner.trace['actions'])
            if recovery:recovery_records[h]['teacher_recovery_end']=len(histories[h])
        runner.cap=500
        for h,history in histories.items():
            reference=read(folder/(h+'_DISCOVERY.json'))
            for q,chain in [('C0',['terminal']),('C_A',['anchor_A','terminal']),('C_B',['anchor_B','terminal'])]:
                current_path=folder/(h+'__'+q+'.json');runner.start(proposal['start'],selected_yaw)
                for action in history:runner.move(action)
                # Same-condition raw replay, no midpoint assignment or tolerance.
                for actual,expected in zip(runner.trace['observations'],reference['observations']):
                    if actual['rgb_hash']!=expected['rgb_hash'] or actual['semantic_hash']!=expected['semantic_hash']:
                        raise ValueError('HISTORY_RAW_REPLAY_MISMATCH')
                cutoff=len(history)
                for role in chain:runner.witness(role,targets[role])
                runner.move('S');runner.trace['complete']=True;runner.trace['cutoff']=cutoff
                runner.save(current_path)
                labels={task:evaluate(compiler,runner.trace,task,cutoff) for task in ('task_A','task_B','task_T')}
                if any(r['safe_v16_label']=='UNKNOWN' for r in labels.values()):raise Rejected('UNKNOWN_CROSS_LABEL')
                traces[h+'__'+q]=dict(path=str(current_path.relative_to(LINE)),sha256=sha(current_path),labels=labels)
        # Both positive and absent-event negatives must physically exist.
        for h in histories:
            role='task_A' if h.startswith('H_A') else 'task_B';other='task_B' if role=='task_A' else 'task_A'
            if traces[h+'__C0']['labels'][role]['safe_v16_label']!='PASS' or traces[h+'__C0']['labels'][other]['safe_v16_label']!='FAIL':raise Rejected('MISSING_REQUIRED_HISTORY_SENSITIVE_LABELS')
            for task in ('task_A','task_B','task_T'):
                if not any(traces[h+'__'+q]['labels'][task]['safe_v16_label']=='PASS' for q in ('C0','C_A','C_B')):raise Rejected('NO_REAL_PASS_TEACHER')
        family=dict(family_id=proposal['id'],house=house['house'],split=house['split'],scene=house['scene'],roles=roles,
            compiler=compiler_config,terminal_instruction=terminal_instruction(roles,house['split']),histories=histories,
            initial_position=proposal['start'],initial_yaw=selected_yaw,traces=traces,recovery=recovery_records,
            content_root=str(store.root.relative_to(LINE)),proposal=proposal,
            shared_cross_history_window_claim=False,training_admission='V16_PHYSICAL_AND_LABEL_CERTIFIED_PENDING_SPLIT_AUDIT')
        write(folder/'FAMILY.json',family,True)
        return family
    except BaseException as exc:
        if current_path and not current_path.exists() and hasattr(runner,'trace'):runner.save(current_path)
        write(folder/'FAILURE.json',dict(error=repr(exc),physical_attempt=True),True)
        if isinstance(exc,Rejected):return None
        raise

def main(run, splits=('FIT','DEV')):
    cfg=read(run/'PROTOCOL.json');manifest=read(HERE/'DATA_MANIFEST.json')
    runtime=LINE/'data_pipeline/mechanism_runtime_v1'
    sys.path.insert(0,str(runtime))
    backend_module=load('v16_real_backend',runtime/'habitat_backend.py')
    feedback=load('v16_real_target_proposals',runtime/'feedback_generation_v1/feedback.py')
    store=ContentStore(run/'content',cfg['artifact_gib']*2**30)
    families=[];began=time.monotonic();shortfalls=[]
    for house in manifest['houses']:
        if house['split'] not in splits:continue
        housefile=run/('HOUSE_'+house['house']+'.json')
        if housefile.exists():
            record=read(housefile);families.extend(record['families'])
            if not record['complete']:shortfalls.append(dict(house=house['house'],actual=len(record['families']),target=house['target_families']))
            continue
        roles={'role_'+str(i):row['spec'] for i,row in enumerate(house['roles'])}
        planner=load('v16_reserved_metadata_check',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
        objects=planner.parse_house(Path(house['scene']).with_suffix('.house').read_text())
        reserved=[o for idx,o in objects.items() if idx in backend_module.RESERVED_MASKS]
        excluded={key:spec for key,spec in roles.items() if any(backend_module.role_matches(o,spec) for o in reserved)}
        immutable(run/('RESERVED_ROLE_EXCLUSIONS_'+house['house']+'.json'),dict(excluded=excluded,rule='exclude whole role signature when a reserved mask would satisfy it; never reinterpret reserved semantic pixels'))
        roles={k:v for k,v in roles.items() if k not in excluded}
        backend=backend_module.HabitatBackend(house['scene'],cfg['gpu'],roles,store,dict(runtime_allowed=True,scene_glb=house['scene'],gpu_device=cfg['gpu']))
        prior=run/('PRIOR_HOUSE_'+house['house']+'.json')
        accepted=list(read(prior)['families']) if prior.exists() else [];attempts=0
        try:
            expected={k:house['roles'][int(k[5:])]['eligible'] for k in roles}
            if backend.eligible!=expected:raise ValueError('SEMANTIC_ASSET_IDENTITY')
            backend.sim.pathfinder.seed(1209)
            starts=[backend.sim.pathfinder.get_random_navigable_point().tolist() for _ in range(cfg['collection']['hubs_per_house'])]
            proposals=[]
            for hub,start in enumerate(starts):
                if not all(math.isfinite(x) for x in start):continue
                targets=[]
                for name in roles:
                    i=int(name[5:])
                    target=feedback.candidate_targets(backend,start,'role_'+str(i),limit=cfg['collection']['targets_per_role'])
                    if target:targets.append((target[0]['distance'],i,target))
                targets.sort(key=lambda x:(x[0],x[1]));targets=targets[:cfg['collection']['roles_per_hub']]
                for triple in itertools.combinations(targets,3):
                    indices=[x[1] for x in triple]
                    if len({house['roles'][i]['signature'][1] for i in indices})<3:continue
                    for terminal in range(3):
                        order=[i for i in range(3) if i!=terminal]+[terminal]
                        p=dict(start=start,hub=hub,role_indices=[indices[i] for i in order],targets=[triple[i][2] for i in order])
                        p['id']='V16_'+digest([house['house'],p])[:20]
                        proposals.append(p)
            seen=set();ancestor=run
            while (ancestor/'REUSED_PHYSICAL_ASSETS.json').exists():
                ancestor=Path(read(ancestor/'REUSED_PHYSICAL_ASSETS.json')['source'])
                seen.update(p.name for p in (ancestor/'proposals').glob('*'))
            skipped=[p['id'] for p in proposals if p['id'] in seen]
            proposals=[p for p in proposals if p['id'] not in seen]
            immutable(run/('PRIOR_PROPOSAL_EXCLUSIONS_'+house['house']+'.json'),dict(skipped=skipped,prior_attempts=len(seen)))
            ordered=[]
            for hub in range(len(starts)):
                group=sorted([p for p in proposals if p['hub']==hub],key=lambda p:(p['targets'][2][0]['distance'],sum(t[0]['distance'] for t in p['targets']),p['id']))
                for i,p in enumerate(group):ordered.append((i,hub,p))
            proposals=[p for _,_,p in sorted(ordered,key=lambda x:x[:2])]
            proposals=proposals[:cfg['collection']['max_proposals_per_house']]
            immutable(run/('PROPOSALS_'+house['house']+'.json'),proposals)
            for proposal in proposals:
                path=run/'proposals'/proposal['id'];attempts+=1
                if (path/'FAMILY.json').exists():family=read(path/'FAMILY.json')
                elif (path/'FAILURE.json').exists():continue
                else:
                    append(run/'COLLECTION_ATTEMPTS.jsonl',dict(status='START',house=house['house'],split=house['split'],proposal=proposal['id'],unix=time.time()))
                    family=collect_family(backend,proposal,house,run,cfg['collection'],store)
                    append(run/'COLLECTION_ATTEMPTS.jsonl',dict(status='CERTIFIED' if family else 'REJECTED',proposal=proposal['id'],unix=time.time()))
                if family:accepted.append(family)
                write(run/'COLLECTION_PROGRESS.json',dict(house=house['house'],split=house['split'],attempts=attempts,accepted=len(accepted),target=house['target_families'],seconds=time.monotonic()-began))
                print(house['house'],attempts,len(accepted),flush=True)
                if len(accepted)==house['target_families']:break
            write(housefile,dict(house=house['house'],families=accepted,attempts=attempts,target=house['target_families'],complete=len(accepted)==house['target_families']),True)
            families.extend(accepted)
            if len(accepted)<house['target_families']:shortfalls.append(dict(house=house['house'],actual=len(accepted),target=house['target_families']))
        finally:backend.close()
    if shortfalls:
        write(run/'DATA_SHORTFALL.json',dict(houses=shortfalls,accepted_families=len(families)),True)
        raise RuntimeError('REGISTERED_HOUSE_FAMILY_SHORTFALL: '+str(shortfalls))
    return families

if __name__=='__main__':
    run=Path(sys.argv[1]);main(run,tuple(sys.argv[2:]) or ('FIT','DEV'))
