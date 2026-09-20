"""Privileged SEE2 service. Only instruction and RGB are returned to the actor."""
import base64
import json
from pathlib import Path
import socket
import sys
import traceback

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15'))
from common import c,load,exact_pose_equal
from content_store import AuditedContentStore,sealed
sealed.FEEDBACK_ROOT=HERE
# This service writes in the new authorized research run directory. Retain the
# inherited exact-path, symlink, owner and content checks within that directory.
sys.modules['guard'].RUNTIME_ROOT=HERE
backend_module=load('v15_continuation_backend',LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py')
compiler_module=load('v15_continuation_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')


class NoInteriorJoin(backend_module.HabitatBackend):
    def reconstruct(self,pose):
        raise RuntimeError('INTERIOR_STATE_ASSIGNMENT_FORBIDDEN')

    def initial(self,position,rotation):
        # Same initialization as the executed raw_run_008 calibrate_start.py.
        self.sim.seed(0)
        state=self.hs.AgentState()
        state.position=self.np.asarray(position,dtype=self.np.float32)
        state.rotation=self.np.quaternion(*rotation)
        self.sim.initialize_agent(0,state)
        self.counts['trace_initializations']+=1
        self._obs=self.sim.reset()
        self.counts['explicit_resets']+=1


def main(fd,run):
    protocol=c.read(HERE/'CONTINUATION_PROTOCOL.json')
    families={f['family_id']:f for f in c.read(LINE/protocol['raw_config'])['families']}
    sock=socket.socket(fileno=fd);stream=sock.makefile('rw');backend=None;active=None;completed=0
    def send(payload):
        stream.write(json.dumps(payload)+'\n');stream.flush()
    def observe():
        observation=dict(backend.observe(),step=len(trace['observations']))
        trace['observations'].append(observation)
        return observation
    def check_prefix():
        t=len(trace['actions']);actual=trace['observations'][-1];expected=reference['observations'][t]
        if (actual['rgb_hash']!=expected['rgb_hash'] or actual['semantic_hash']!=expected['semantic_hash'] or
                not exact_pose_equal(actual['pose'],expected['pose'])):
            c.write(folder/'PREFIX_DIVERGENCE.json',dict(step=t,actual=actual,expected=expected),True)
        assert actual['rgb_hash']==expected['rgb_hash'],'PREFIX_RAW_RGB_MISMATCH'
        assert actual['semantic_hash']==expected['semantic_hash'],'PREFIX_AUDIT_EVIDENCE_MISMATCH'
        assert exact_pose_equal(actual['pose'],expected['pose']),'PREFIX_PHYSICAL_STATE_MISMATCH'
    def picture(reset=False):
        rgb=backend.np.ascontiguousarray(backend._obs['rgb'][:,:,:3]).tobytes()
        message=dict(done=False,rgb=base64.b64encode(rgb).decode())
        if reset:message['instruction']=compiler.tasks[condition['task_id']]['instruction']
        send(message)
    try:
        with AuditedContentStore(run/'content',20*2**30) as store:
            for line in stream:
                request=json.loads(line)
                if request=={'op':'close'}:
                    assert active is None and completed==len(protocol['rollouts'])
                    break
                if request.get('op')=='reset':
                    assert active is None and set(request)=={'op','rank'} and request['rank']==completed
                    active=protocol['rollouts'][completed];condition=protocol['conditions'][active['condition']]
                    family=families[condition['family_id']]
                    folder=run/'rollouts'/f'{completed:03d}'
                    compiler=compiler_module.Compiler(**family['compiler'])
                    if backend:backend.close()
                    backend=NoInteriorJoin(family['scene'],protocol['gpu'],family['roles'],store,
                        dict(runtime_allowed=True,scene_glb=family['scene'],gpu_device=protocol['gpu']))
                    assert backend.eligible==family['compiler']['eligible']
                    candidate=family['candidate'];history=candidate['histories'][condition['history_id']]
                    reference=c.read(LINE/protocol['raw_run']/family['family_id']/(condition['history_id']+'__C0.json'))
                    start=family['initial_states'][condition['history_id']]
                    backend.initial(start['position'],start['rotation'])
                    trace=dict(actions=[],observations=[],collisions=0,complete=False,interior_state_assignments=0)
                    observe();check_prefix();picture(reset=True)
                    continue
                assert active is not None and set(request)=={'op','action'} and request['op']=='action'
                action=request['action'];assert action in c.ACTIONS
                code='FLRS'[c.ACTIONS.index(action)];before=len(trace['actions'])
                assert before<500
                if before<len(history):assert code==history[before],'FIXED_HISTORY_ACTION_MISMATCH'
                trace['actions'].append(code)
                if code!='S':
                    trace['collisions']+=int(backend.step(code));observe()
                if len(trace['actions'])<=len(history):check_prefix()
                done=code=='S' or len(trace['actions'])==500
                if not done:
                    picture();continue
                assert before>=len(history),'PREFIX_TERMINATED_EARLY'
                trace['complete']=True
                assert backend.counts['explicit_reconstructions']==0
                label=compiler.evaluate(trace,condition['task_id'])
                events=compiler.atoms(trace['observations']);task=compiler.tasks[condition['task_id']]
                c.write(folder/'TRACE_PRIVILEGED.json',trace,True)
                c.write(folder/'TASK_RESULT.json',dict(rank=completed,condition=condition,model=active['model'],
                    label=label,stopped=code=='S',total_decisions=len(trace['actions']),
                    prefix_decisions=len(history),autonomous_decisions=len(trace['actions'])-len(history),
                    collisions=trace['collisions'],evidence_complete=all(o['evidence_complete'] for o in trace['observations']),
                    prefix_replayed_and_exact=True,interior_state_assignments=0,
                    terminal_witness_at_stop=bool(events[-1][task['terminal']]),
                    earlier_anchor_witness=any(bool(e[task['anchor']]) for e in events[:-1]),
                    frozen_compiler_unknown_on_collision=trace['collisions']>0,
                    scope='Autonomous continuation after a fixed real history; not R2R SR'),True)
                completed+=1;active=None;send(dict(done=True))
            audit=store.full_audit();assert audit['audit_pass']
            c.write(run/'CONTENT_AUDIT.json',audit,True)
            c.write(run/'SERVICE_RESULT.json',dict(completed=completed,all_prefixes_exact=True),True)
        send(dict(closed=True))
    except BaseException as exc:
        c.write(run/'SERVICE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
    finally:
        if backend:backend.close()
        stream.close();sock.close()


if __name__=='__main__':
    run=Path(sys.argv[2])
    try:main(int(sys.argv[1]),run)
    except BaseException as exc:
        if not (run/'SERVICE_FAILURE.json').exists():
            c.write(run/'SERVICE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
