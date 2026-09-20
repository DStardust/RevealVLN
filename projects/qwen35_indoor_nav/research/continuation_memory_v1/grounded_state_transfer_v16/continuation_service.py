"""Private physical service: full history replay, no oracle fields returned to actor."""
import base64
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from collect import ContentStore
from evaluator_v16 import legacy,evaluate
backend_module=load('v16_live_habitat',LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py')

class NoInteriorJoin(backend_module.HabitatBackend):
    def reconstruct(self,pose):raise ValueError('INTERIOR_STATE_ASSIGNMENT_FORBIDDEN')

def exact_pose(a,b):
    return a['position']==b['position'] and (a['rotation']==b['rotation'] or a['rotation']==[-x for x in b['rotation']]) and a.get('sensors',{}).keys()==b.get('sensors',{}).keys() and all(exact_pose(a['sensors'][k],b['sensors'][k]) for k in a.get('sensors',{}))

def main(fd,session):
    config=read(session/'CONFIG.json');run=Path(config['run']);registry=read(run/'EVALUATION_REGISTRY.json')
    families={f['family_id']:f for f in read(run/'DATA.json')['raw_families']}
    sock=socket.socket(fileno=fd);stream=sock.makefile('rw');backend=None;backend_family=None;active=None;done=0
    store=ContentStore(session/'content',config['artifact_gib']*2**30)
    def send(value):stream.write(json.dumps(value)+'\n');stream.flush()
    def observe():trace['observations'].append(dict(backend.observe(),step=len(trace['observations'])))
    def check_prefix():
        t=len(trace['actions']);a=trace['observations'][-1];b=reference['observations'][t]
        if a['rgb_hash']!=b['rgb_hash'] or a['semantic_hash']!=b['semantic_hash'] or not exact_pose(a['pose'],b['pose']):
            write(folder/'PREFIX_DIVERGENCE.json',dict(step=t,actual=a,expected=b),True)
            raise ValueError('REGISTERED_PHYSICAL_PREFIX_MISMATCH')
    def picture(reset=False):
        value=dict(done=False,rgb=base64.b64encode(backend.np.ascontiguousarray(backend._obs['rgb'][:,:,:3]).tobytes()).decode())
        if reset:value['instruction']=family['terminal_instruction'] if condition['task_id']=='task_T' else compiler.tasks[condition['task_id']]['instruction']
        send(value)
    try:
        for line in stream:
            request=json.loads(line)
            if request==dict(op='close'):
                if active is not None:raise ValueError('ACTIVE_ROLLOUT_AT_CLOSE')
                send(dict(closed=True));break
            if request.get('op')=='reset':
                if active is not None:raise ValueError('UNFINISHED_ROLLOUT')
                rank=request['rank'];slot=registry['slots'][rank]
                if rank not in config['ranks']:raise ValueError('UNREGISTERED_SLOT')
                condition=registry['conditions'][slot['condition']];family=families[condition['family_id']]
                folder=session/'rollouts'/f'{rank:04d}';folder.mkdir(parents=True,exist_ok=True)
                compiler=legacy.Compiler(**family['compiler']);history=family['histories'][condition['history_id']]
                reference=read(LINE/family['traces'][condition['history_id']+'__C0']['path'])
                if backend_family!=family['family_id']:
                    if backend:backend.close()
                    backend=NoInteriorJoin(family['scene'],config['gpu'],family['roles'],store,dict(runtime_allowed=True,scene_glb=family['scene'],gpu_device=config['gpu']))
                    backend_family=family['family_id']
                if backend.eligible!=family['compiler']['eligible']:raise ValueError('RUNTIME_ROLE_IDENTITY')
                backend.reset(family['initial_position'],family['initial_yaw'],0)
                trace=dict(actions=[],observations=[],collisions=0,complete=False,interior_state_assignments=0)
                active=rank;observe();check_prefix();picture(True);continue
            if active is None or set(request)!=set(('op','action')) or request['op']!='action':raise ValueError('SERVICE_PROTOCOL')
            action=request['action']
            if action not in c.ACTIONS:raise ValueError('ACTION_INTERFACE')
            code='FLRS'[c.ACTIONS.index(action)];t=len(trace['actions'])
            if t>=500:raise ValueError('DECISION_BUDGET')
            if t<len(history) and code!=history[t]:raise ValueError('FORCED_HISTORY_CHANGED')
            trace['actions'].append(code)
            if code!='S':trace['collisions']+=int(backend.step(code));observe()
            if len(trace['actions'])<=len(history):check_prefix()
            if code!='S' and len(trace['actions'])<500:picture();continue
            if t<len(history):raise ValueError('EARLY_PREFIX_TERMINATION')
            trace['complete']=True
            if backend.counts['explicit_reconstructions']:raise ValueError('INTERIOR_TELEPORT')
            result=evaluate(compiler,trace,condition['task_id'],len(history))
            result.update(rank=rank,condition=condition,model=slot['model'],prefix_decisions=len(history),
                          autonomous_decisions=len(trace['actions'])-len(history),prefix_replayed_and_exact=True)
            write(folder/'TRACE_PRIVILEGED.json',trace,True);write(folder/'TASK_RESULT.json',result,True)
            done+=1;active=None;send(dict(done=True))
        write(session/'SERVICE_RESULT.json',dict(completed=done,content_bytes=store.bytes,interior_state_assignments=0),True)
    finally:
        if backend:backend.close()
        stream.close();sock.close()

if __name__=='__main__':main(int(sys.argv[1]),Path(sys.argv[2]))
