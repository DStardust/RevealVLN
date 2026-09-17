"""G1F isolated real-trajectory discovery. Metadata is never policy input."""
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time

import habitat_sim as hs
import numpy as np
import quaternion

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ROOT = LINE.parents[1]
RUNTIME = OUT.parent/'Q35N_G0R_DEPENDENCY_RECOVERY_V1'
SCENE = ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d/17DRP5sb8fy'
TRAIN = ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
KINDS = {'D':('chair','dining room'), 'K':('sink','kitchen'),
         'B':('bed','bedroom'), 'L':('tv_monitor','living room')}
TAILS = ['FFFFFFFF','LFFRFFFF','RFFLFFFF','LLFFRRFF','RRFFLLFF','LFRFLFRF','RFLFRFLF','LRLRLRLR']
NAMES = {'F':'move_forward','L':'turn_left','R':'turn_right'}

def digest(x):
    if not isinstance(x,bytes):
        x = json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(x).hexdigest()

def save(name, obj):
    p = OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def log(name,obj):
    with (OUT/name).open('a') as f:
        f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')

def rotations(n):
    n %= 24
    return ['L']*n if n <= 12 else ['R']*(24-n)

def compress(actions):
    out, n = [],0
    for a in actions:
        if a == 'F':
            out += rotations(n)+['F']; n=0
        else:
            assert a in ('L','R')
            n += 1 if a == 'L' else -1
    return out+rotations(n)

def inverse(actions):
    raw=[]
    for a in reversed(actions):
        raw += ['L']*12+['F']+['R']*12 if a=='F' else ['R' if a=='L' else 'L']
    return raw,compress(raw)

def transform(actions):
    p=np.zeros(2); yaw=0
    for a in actions:
        if a=='F': p += .25*np.array([-math.sin(yaw*math.pi/12),-math.cos(yaw*math.pi/12)])
        else: yaw += 1 if a=='L' else -1
    return {'xz':p.tolist(),'yaw_bin':yaw%24}

def pose(state):
    return {'position':state.position.tolist(),'rotation':quaternion.as_float_array(state.rotation).tolist(),
            'sensors':{k:{'position':v.position.tolist(),'rotation':quaternion.as_float_array(v.rotation).tolist()}
                       for k,v in state.sensor_states.items()}}

def yawbin(q):
    v=quaternion.rotate_vectors(q,[0,0,-1])
    return int(round(math.atan2(-v[0],-v[2])*12/math.pi))%24

def pose_spread(a,b):
    p=float(np.linalg.norm(np.array(a['position'])-b['position']))
    q1=np.array(a['rotation']); q2=np.array(b['rotation'])
    # atan2 form retains precision for nearly-identical float32 quaternions.
    q1/=np.linalg.norm(q1); q2/=np.linalg.norm(q2)
    angle=float(4*np.arcsin(min(1.,min(np.linalg.norm(q1-q2),np.linalg.norm(q1+q2))/2)))
    return p,angle

class Reject(Exception):
    def __init__(self,code,**detail):
        self.code,self.detail=code,detail
        super().__init__(code)

class Engine:
    def __init__(self,phase='discovery'):
        self.phase=phase
        self.counts=collections.Counter(); self.started=time.time(); self.route_cache={}
        cfg=hs.SimulatorConfiguration(); cfg.scene_id=str(SCENE/'17DRP5sb8fy.glb')
        cfg.gpu_device_id=2; cfg.enable_physics=False; cfg.allow_sliding=False
        ac=hs.agent.AgentConfiguration(); ac.height=1.5; ac.radius=.1
        specs=[]
        for name,typ in [('rgb',hs.SensorType.COLOR),('semantic',hs.SensorType.SEMANTIC)]:
            s=hs.SensorSpec(); s.uuid=name; s.sensor_type=typ; s.resolution=[224,224]
            s.position=[0,1.25,0]; s.orientation=[0,0,0]; s.parameters['hfov']='90'; s.gpu2gpu_transfer=False
            specs.append(s)
        ac.sensor_specifications=specs
        ac.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=.25 if k=='F' else 15)) for k,name in NAMES.items()}
        self.sim=hs.Simulator(hs.Configuration(cfg,[ac])); self.counts['simulator_constructions']=1
        self.eligible={k:[] for k in KINDS}; self.objects={}
        for obj in self.sim.semantic_scene.objects:
            if obj is None: continue
            idx=int(obj.id.rsplit('_',1)[1]); region=obj.region
            rec={'id':obj.id,'mask_id':idx,'raw':obj.category.name('raw'),
                 'mpcat40':obj.category.name('mpcat40'),
                 'room':region.category.name() if region else None,'region_id':region.id if region else None,
                 'center':obj.obb.center.tolist(),'sizes':obj.obb.sizes.tolist()}
            assert idx not in self.objects
            self.objects[idx]=rec
            for kind,(category,room) in KINDS.items():
                raw=rec['raw'].lower().replace('#',' ')
                rawok=bool(re.search(r'\bchair\b',raw)) if kind=='D' else raw=={'K':'sink','B':'bed','L':'tv'}.get(kind)
                if rec['mpcat40']==category and rec['room']==room and rawok:
                    assert idx>0, 'Eligible object ID overlaps zero clear value'
                    self.eligible[kind].append(idx)
        save('SEMANTIC_INVENTORY.json',{'objects':list(self.objects.values()),'eligible':self.eligible,
             'mapping_basis':'SemanticObject.id terminal field equals .house object_index; native GenericInstanceMeshData preserves semantic PLY objectId attributes',
             'model_recognition_verified':False})
        if not all(self.eligible.values()): raise Reject('ELIGIBLE_SET_EMPTY',eligible=self.eligible)
        self.views={k:[] for k in KINDS}

    def close(self):
        self.sim.close(); save(f'{self.phase}_COUNTS.json',dict(self.counts))

    def start(self,position,bin,seed=0):
        self.sim.seed(seed)
        s=hs.AgentState(); s.position=np.asarray(position,dtype=np.float32)
        a=bin*math.pi/24; s.rotation=np.quaternion(math.cos(a),0,math.sin(a),0)
        self.sim.initialize_agent(0,s); self.counts['trace_initializations']+=1
        obs=self.sim.reset(); self.counts['explicit_resets']+=1
        return obs

    def record(self,obs):
        self.counts['observation_records']+=1
        rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]); sem=np.ascontiguousarray(obs['semantic'])
        assert rgb.shape==(224,224,3) and sem.shape==(224,224)
        state=pose(self.sim.get_agent(0).get_state())
        a,b=state['sensors']['rgb'],state['sensors']['semantic']
        assert pose_spread(a,b)[0]<1e-7 and pose_spread(a,b)[1]<1e-7
        ids,nums=np.unique(sem,return_counts=True); pixels={int(k):int(v) for k,v in zip(ids,nums)}
        unknown=set(pixels)-set(self.objects)-{0,65535,4294967295}
        return {'rgb_hash':digest(rgb.tobytes()),'semantic_hash':digest(sem.tobytes()),
                'pose':state,'pixels':pixels,'evidence_complete':not unknown,
                'unknown_mask_ids':sorted(unknown)},rgb,sem

    def trace(self,position,bin,actions,seed=0,keep=False,tag='discovery'):
        obs=self.start(position,bin,seed); records=[]; events=[]; blobs=[]
        actions=list(actions); collisions=0
        for t in range(len(actions)+1):
            r,rgb,sem=self.record(obs); r['step']=t
            ev={k:[] for k in KINDS}
            if records and r['evidence_complete'] and records[-1]['evidence_complete']:
                for k,indices in self.eligible.items():
                    ev[k]=[idx for idx in indices if r['pixels'].get(idx,0)>=256 and records[-1]['pixels'].get(idx,0)>=256]
            r['see2']=ev; records.append(r)
            if keep:
                for arr,h,suffix in [(rgb,r['rgb_hash'],'rgb'),(sem,r['semantic_hash'],'semantic')]:
                    p=OUT/'content'/f'{h}.{suffix}.npy'
                    if not p.exists():
                        p.parent.mkdir(exist_ok=True); np.save(p,arr,allow_pickle=False)
            if t==len(actions): break
            if actions[t]=='S':
                assert t==len(actions)-1
                self.counts['executed_stops']+=1
                break
            obs=self.sim.step(NAMES[actions[t]]); self.counts['primitive_actions']+=1
            self.counts['discovery_primitive_actions' if tag=='discovery' else 'certification_primitive_actions']+=1
            if obs['collided']:
                collisions+=1; self.counts['collisions']+=1
                break
        out={'actions':actions,'observations':records,'collisions':collisions,
             'complete':len(records)==len(actions)+(0 if actions and actions[-1]=='S' else 1) and not collisions,
             'seed':seed,'initial_position':list(map(float,position)),'initial_yaw_bin':bin}
        out['trace_hash']=digest(out)
        self.counts['actual_trace_replays']+=1
        return out

    def preview(self):
        # Initializing viewpoint proposals does NOT establish reachability from u.
        for kind,ids in self.eligible.items():
            for idx in sorted(ids):
                c=np.array(self.objects[idx]['center']); proposed=[]
                for angle in range(8):
                    for radius in (.75,1.25):
                        p=c+np.array([radius*math.cos(angle*math.pi/4),0,radius*math.sin(angle*math.pi/4)])
                        proposed.append((tuple(p.tolist()),angle,radius))
                seen=set()
                for original,angle,radius in sorted(proposed):
                    p=self.sim.pathfinder.snap_point(np.array(original,dtype=np.float32))
                    entry={'kind':kind,'instance':idx,'original':list(original),'angle':angle,'radius':radius}
                    if not np.isfinite(p).all():
                        log('WITNESS_DISCOVERY.jsonl',dict(entry,status='REJECTED',reason='NONFINITE_SNAP')); continue
                    key=tuple(p.tolist())
                    if key in seen:
                        log('WITNESS_DISCOVERY.jsonl',dict(entry,status='REJECTED',reason='DUPLICATE_SNAP')); continue
                    seen.add(key)
                    target=int(round(math.atan2(-(c-p)[0],-(c-p)[2])*12/math.pi))%24
                    tr=self.trace(p,target,['L','R'])
                    good=tr['complete'] and all(o['evidence_complete'] for o in tr['observations']) and any(idx in o['see2'][kind] for o in tr['observations'])
                    entry.update(position=p.tolist(),yaw_bin=target,trace_hash=tr['trace_hash'],
                                 pixel_counts=[o['pixels'].get(idx,0) for o in tr['observations']],
                                 status='WITNESS_PREVIEW_PASS' if good else 'REJECTED',reason=None if good else 'SEE2_UNSUPPORTED')
                    log('WITNESS_DISCOVERY.jsonl',entry)
                    log('PREVIEW_TRACES.jsonl',tr)
                    if good: self.views[kind].append(entry)
        save('WITNESS_POOL.json',self.views)
        if not all(self.views.values()): raise Reject('WITNESS_POOL_EMPTY',counts={k:len(v) for k,v in self.views.items()})

    def u_pool(self):
        assert digest(TRAIN.read_bytes())=='f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
        with gzip.open(TRAIN,'rt') as f: episodes=json.load(f)['episodes']
        seen=set(); pool=[]
        for ep in sorted(episodes,key=lambda e:int(e['episode_id'])):
            if '17DRP5sb8fy' not in ep['scene_id']: continue
            for j,p in enumerate([ep['start_position']]+ep.get('reference_path',[])):
                key=tuple(p)
                if key in seen:continue
                seen.add(key); snapped=self.sim.pathfinder.snap_point(np.asarray(p,dtype=np.float32))
                pool.append({'episode_id':ep['episode_id'],'path_index':j-1,'original':p,
                             'position':snapped.tolist() if np.isfinite(snapped).all() else None})
                if len(pool)==32:break
            if len(pool)==32:break
        save('U_POOL.json',pool)
        return pool

    def plan(self,position,bin,view):
        key=digest([list(map(float,position)),bin,view])
        if key in self.route_cache:return self.route_cache[key]
        self.start(position,bin)
        follower=hs.GreedyGeodesicFollower(self.sim.pathfinder,self.sim.get_agent(0),goal_radius=.1875,fix_thrashing=True)
        try:
            seq=follower.find_path(np.asarray(view['position'],dtype=np.float32))
            mapping={v:k for k,v in NAMES.items()}
            actions=[mapping[a] for a in seq if a is not None]
            if len(actions)>504:raise Reject('DISCOVERY_HISTORY_BUDGET_REJECT',planned=len(actions))
            tr=self.trace(position,bin,actions)
            if not tr['complete']:raise Reject('ROUTE_COLLISION')
            state=self.sim.get_agent(0).get_state()
            center=np.asarray(self.objects[view['instance']]['center']); delta=center-state.position
            target=int(round(math.atan2(-delta[0],-delta[2])*12/math.pi))%24
            actions+=rotations(target-yawbin(state.rotation))+['L','R']
            tr=self.trace(position,bin,actions)
            if not tr['complete']:raise Reject('ROUTE_COLLISION')
            if not any(view['instance'] in o['see2'][view['kind']] for o in tr['observations']):raise Reject('ACTUAL_WITNESS_UNSUPPORTED')
            val=(actions,tr,None)
        except Reject as e: val=(None,None,{'code':e.code,**e.detail})
        except hs.errors.GreedyFollowerError:val=(None,None,{'code':'GREEDY_PATH_FAILURE'})
        self.route_cache[key]=val
        log('ROUTE_PLANS.jsonl',{'key':key,'view':view,'actions':val[0],'failure':val[2],
                               'trace':val[1]})
        return val

    @staticmethod
    def pattern(tr,required,forbidden):
        if not tr['complete'] or not all(o['evidence_complete'] for o in tr['observations']):return False
        observed={k for o in tr['observations'] for k,idx in o['see2'].items() if idx}
        return set(required)<=observed and not set(forbidden)&observed

    def loop(self,position,bin,kind,forbidden):
        failures=[]
        for view in self.views[kind]:
            acts,tr,error=self.plan(position,bin,view)
            if error: failures.append(error);continue
            if not self.pattern(tr,[kind],forbidden):
                failures.append({'code':'OUTBOUND_FORBIDDEN_EVENT_OR_INCOMPLETE_PRUNE'})
                continue
            raw,inv=inverse(acts); final=acts+inv
            if len(final)>504:
                failures.append({'code':'DISCOVERY_HISTORY_BUDGET_REJECT','count':len(final)});continue
            # Both versions must preserve ideal transform AND survive actual replay.
            a,b=transform(raw),transform(inv)
            assert a['yaw_bin']==b['yaw_bin'] and np.linalg.norm(np.array(a['xz'])-b['xz'])<1e-8
            uncompressed=self.trace(position,bin,acts+raw)
            compressed=self.trace(position,bin,final)
            log('LOOP_REPLAYS.jsonl',{'kind':kind,'view':view,'raw_inverse':raw,'compressed_inverse':inv,
                                   'raw_transform':a,'compressed_transform':b,
                                   'raw_trace':uncompressed,'compressed_trace':compressed})
            if not self.pattern(compressed,[kind],forbidden) or not uncompressed['complete']:
                failures.append({'code':'LOOP_EVENT_OR_LEGALITY_REJECT'});continue
            p,r=pose_spread(compressed['observations'][0]['pose'],compressed['observations'][-1]['pose'])
            if p>1e-4 or r>1e-4:
                failures.append({'code':'LOOP_POSE_DRIFT','position_m':p,'angle_rad':r});continue
            return final,compressed,view,failures
        raise Reject('NO_VALID_'+kind+'_LOOP',subfailures=dict(collections.Counter(f['code'] for f in failures)),examples=failures[:5])

    def candidate_base(self,position,bin):
        d,dt,dv,_=self.loop(position,bin,'D',['K'])
        k,kt,kv,_=self.loop(position,bin,'K',['D'])
        l,lt,lv,_=self.loop(position,bin,'L',['D','K','B'])
        histories={'H_D':d,'H_K':k,'H_D_L':d+l}
        n=max(map(len,histories.values()))
        if n+8>512:raise Reject('DISCOVERY_HISTORY_BUDGET_REJECT',lengths={k:len(v) for k,v in histories.items()})
        starts=dt['observations'][0]
        # All candidates use the same deterministic first event-neutral 2-action loop.
        neutral=None
        for pad in (['L','R'],['R','L'],['L']*24,['R']*24):
            pt=self.trace(position,bin,pad)
            if self.pattern(pt,[],['D','K','B']):neutral=pad;break
        if neutral is None:raise Reject('NO_NEUTRAL_PADDING')
        results={}
        for h,a in histories.items():
            needed=n-len(a)
            if needed%len(neutral):raise Reject('PADDING_PARITY_REJECT')
            a=a+neutral*(needed//len(neutral))
            tr=self.trace(position,bin,a)
            required=['D'] if h!='H_K' else ['K']
            if h=='H_D_L': required+=['L']
            forbidden=['K'] if h!='H_K' else ['D']
            if not self.pattern(tr,required,forbidden):raise Reject('HISTORY_EVENT_PATTERN_REJECT',history=h)
            results[h]={'actions':a,'trace':tr}
        last=[v['trace']['observations'][-1] for v in results.values()]
        spread=[pose_spread(last[0]['pose'],o['pose']) for o in last]
        hash_equal=all((o['rgb_hash'],o['semantic_hash'])==(last[0]['rgb_hash'],last[0]['semantic_hash']) for o in last)
        detail={'position_spread_m':max(x[0] for x in spread),'rotation_spread_rad':max(x[1] for x in spread),
                'hash_equal':hash_equal,'end_hashes':[(o['rgb_hash'],o['semantic_hash']) for o in last]}
        log('MERGE_PRECHECKS.jsonl',{'position':list(map(float,position)),'yaw_bin':bin,**detail})
        if not hash_equal or detail['position_spread_m']>1e-4 or detail['rotation_spread_rad']>1e-4:
            raise Reject('U_MERGE_REJECT',**detail)
        return results

    def discovery(self):
        pool=self.u_pool(); basecache={}; attempts=0
        ledger=OUT/'DISCOVERY_ATTEMPTS.jsonl'
        completed=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        assert [x['attempt'] for x in completed]==list(range(1,len(completed)+1))
        resume=len(completed)
        for u in pool:
            for bin in range(24):
                key=digest([u,bin])
                for tail in TAILS:
                    if attempts==256:
                        raise Reject('DISCOVERY_ATTEMPTS_EXHAUSTED',attempts=attempts)
                    attempts+=1; self.counts['family_candidates']=attempts
                    if attempts<=resume:continue
                    self.counts['family_candidate_evaluations']+=1
                    try:
                        if u['position'] is None:raise Reject('U_NONFINITE_SNAP')
                        if key not in basecache:
                            try:basecache[key]=(self.candidate_base(u['position'],bin),None)
                            except Reject as e:basecache[key]=(None,e)
                        base,error=basecache[key]
                        if error:raise error
                        # Do not freeze until continuations and all prechecks exist.
                        result=self.complete_candidate(u,bin,base,tail)
                        save('FROZEN_CANDIDATE.json',result)
                        return result
                    except Reject as e:
                        log('DISCOVERY_ATTEMPTS.jsonl',{'attempt':attempts,'u':u,'yaw_bin':bin,'tail':tail,
                            'status':'REJECTED','reason':e.code,'detail':e.detail,'base_cache_key':key})
                    if attempts%8==0:
                        save('PROGRESS.json',{'stage':'discovery','attempts':attempts,'last_reason':e.code if False else 'see ledger',
                             'counts':dict(self.counts),'elapsed_seconds':time.time()-self.started})
                        print(f'Discovery {attempts}/256',flush=True)
        raise Reject('DISCOVERY_POOL_EXHAUSTED',attempts=attempts)

    def complete_candidate(self,u,bin,base,tail):
        histories={}; finals=[]
        for h,data in base.items():
            acts=data['actions']+list(tail)
            tr=self.trace(u['position'],bin,acts)
            if not self.pattern(tr,['K'] if h=='H_K' else ['D'],['D'] if h=='H_K' else ['K']):
                raise Reject('PUBLIC_TAIL_LEGALITY_OR_EVENTS')
            if any(o['see2'][k] for o in tr['observations'][-8:] for k in ('D','K','B')):
                raise Reject('PUBLIC_TAIL_EVENT_REJECT')
            histories[h]=acts; finals.append(tr['observations'][-1])
        if len({(o['rgb_hash'],o['semantic_hash']) for o in finals})!=1:
            raise Reject('S_MERGE_HASH_REJECT')
        if any(max(pose_spread(finals[0]['pose'],o['pose']))>1e-4 for o in finals):
            raise Reject('S_MERGE_POSE_REJECT')
        state=finals[0]['pose']; position=state['position']; q=np.quaternion(*state['rotation']); sb=yawbin(q)
        continuations={}; queries={}
        for cname,anchor,forbidden in [('C0',None,['D','K']),('C_D','D',['K']),('C_K','K',['D'])]:
            options=[None] if anchor is None else self.views[anchor]
            found=None
            for view in options:
                prefix=[]; current_pos=position; current_bin=sb
                if view:
                    prefix,tr,err=self.plan(position,sb,view)
                    if err:continue
                    end=tr['observations'][-1]['pose'];current_pos=end['position']; current_bin=yawbin(np.quaternion(*end['rotation']))
                for bed in self.views['B']:
                    suffix,tr,err=self.plan(current_pos,current_bin,bed)
                    if err:continue
                    acts=prefix+suffix+['S']
                    if len(acts)>160:continue
                    full=self.trace(position,sb,acts)
                    if not self.pattern(full,['B']+([anchor] if anchor else []),forbidden):continue
                    if not full['observations'][-1]['see2']['B']:continue
                    query=make_query(full)
                    if len(query['sequence'])>160:continue
                    found=(acts,full,query);break
                if found:break
            if found is None:raise Reject('NO_VALID_CONTINUATION',continuation=cname)
            continuations[cname]=found[0];queries[cname]=found[2]
        # Complete actual concatenated traces must already match the prospective matrix.
        checks=[]
        for h,a in histories.items():
            for c,b in continuations.items():
                tr=self.trace(u['position'],bin,a+b)
                if not tr['complete']:raise Reject('CONCATENATED_TRACE_REJECT')
                start=tr['observations'][len(a)]
                if (start['rgb_hash'],start['semantic_hash'])!=(finals[0]['rgb_hash'],finals[0]['semantic_hash']):
                    raise Reject('CONCATENATED_MERGE_REJECT')
                for task in ('g_D_v2','g_K_v2'):
                    y=task_eval(tr,task)
                    if y!=expected(task,h,c):raise Reject('PREFLIGHT_MATRIX_MISMATCH',task=task,history=h,continuation=c,outcome=y)
                    checks.append({'task':task,'history':h,'continuation':c,'outcome':y,'trace_hash':tr['trace_hash']})
        candidate={'family_id':'F17_DK_B_v2','u':u,'yaw_bin':bin,'histories':histories,
                   'continuations':continuations,'queries':queries,'public_tail':tail,
                   'merge_observation':finals[0],'preflight_checks':checks}
        candidate['frozen_candidate_hash']=digest(candidate)
        return candidate

def task_eval(trace,task):
    if not trace['complete'] or not all(o['evidence_complete'] for o in trace['observations']):return 'unknown'
    anchor='D' if task=='g_D_v2' else 'K'
    obs=trace['observations']; t=len(obs)-1
    passed=trace['actions'][-1:] == ['S'] and bool(obs[-1]['see2']['B']) and any(o['see2'][anchor] for o in obs[:t])
    return 'pass' if passed else 'fail'

def expected(task,h,c):
    if task=='g_D_v2':return 'pass' if h!='H_K' or c=='C_D' else 'fail'
    return 'pass' if h=='H_K' or c=='C_K' else 'fail'

def make_query(trace):
    seq=[]
    for i,a in enumerate(trace['actions']):
        if i>0:
            for k,indices in trace['observations'][i]['see2'].items():
                if indices:
                    cat,room=KINDS[k]
                    seq.append({'kind':'observe','object_category':cat,'room_category':room,
                                'min_pixels':256,'consecutive_frames':2})
        if a=='S':seq.append({'kind':'act','action':'STOP'})
        else:
            name={'F':'MOVE_FORWARD','L':'TURN_LEFT','R':'TURN_RIGHT'}[a]
            if seq and seq[-1]['kind']=='movement' and seq[-1]['action']==name:seq[-1]['repeat']+=1
            else:seq.append({'kind':'movement','action':name,'repeat':1})
    for i,x in enumerate(seq):x['order']=i
    return {'query_schema_version':'q35n.continuation_query.v2','coordinate_frame':'agent_relative_discrete','sequence':seq,
            'action_trace_ref':'sha256:'+digest(trace['actions']),
            'rgb_content_refs':['sha256:'+o['rgb_hash'] for o in trace['observations']],
            'canonicalization':'semantic_sequence_only_sorted_json_keys_utf8_controlled_vocab_v2'}
