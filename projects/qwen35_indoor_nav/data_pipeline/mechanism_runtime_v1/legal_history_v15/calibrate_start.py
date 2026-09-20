"""Bounded calibration of initial states, with all subsequent motion executed."""
import math
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LINE, RUNTIME, c, load, raw_window, pose_delta,exact_pose_equal
from content_store import AuditedContentStore
from task_controls import terminal_only,neutral_span


def main(run):
    config = c.read(run/'CONFIG.json')
    for path, digest in config['source_hashes'].items():
        assert c.sha(LINE/path) == digest, path
    module = load('v15_start_backend', RUNTIME/'habitat_backend.py')
    compiler_module = load('v15_start_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    class Backend(module.HabitatBackend):
        def reconstruct(self, pose):
            raise RuntimeError('INTERIOR_STATE_ASSIGNMENT_FORBIDDEN')

        def initial(self, position, rotation):
            self.sim.seed(config['environment_seed'])
            state = self.hs.AgentState()
            state.position = self.np.asarray(position, dtype=self.np.float32)
            state.rotation = self.np.quaternion(*rotation)
            self.sim.initialize_agent(0, state)
            self.counts['trace_initializations'] += 1
            self._obs = self.sim.reset()
            self.counts['explicit_resets'] += 1

    decisions = 0
    results = []
    backend = None
    def execute(position, rotation, actions):
        nonlocal decisions
        assert len(actions) <= 500
        backend.initial(position, rotation)
        trace = dict(actions=[], observations=[dict(backend.observe(), step=0)],
                     collisions=0, complete=True, interior_state_assignments=0)
        for action in actions:
            assert decisions < config['max_decisions'], 'DECISION_LIMIT'
            if action != 'S':
                trace['collisions'] += int(backend.step(action))
                trace['observations'].append(dict(backend.observe(),step=len(trace['observations'])))
            trace['actions'].append(action)
            decisions += 1
            if decisions % 20 == 0:
                c.write(run/'PROGRESS.json',dict(unix=time.time(),decisions=decisions))
        return trace

    try:
        with AuditedContentStore(run/'content',config['output_gib']*2**30) as store:
            for family in config['families']:
                folder=run/family['family_id'];folder.mkdir()
                for path,digest in family['assets'].items():assert c.sha(path)==digest,path
                backend=Backend(family['scene'],config['gpu'],family['roles'],store,
                    dict(runtime_allowed=True,scene_glb=family['scene'],gpu_device=config['gpu']))
                compiler=compiler_module.Compiler(**family['compiler'])
                candidate=family['candidate']
                angle=candidate['yaw_bin']*math.pi/24
                origin=list(candidate['position']); rotation=[math.cos(angle),0,math.sin(angle),0]
                prepared=family.get('initial_states')
                if prepared:
                    origin=prepared['H_A']['position'];rotation=prepared['H_A']['rotation']
                reference=execute(origin,rotation,candidate['histories']['H_A'])
                c.write(folder/'REFERENCE.json',reference,True)
                target=reference['observations'][-1]['pose']
                target_window=raw_window(reference,len(reference['actions']))
                starts={};controls={}
                for name,history in candidate['histories'].items():
                    position=list(origin); initial=list(rotation); seen=set(); attempts=[]
                    if prepared:
                        position=prepared[name]['position'];initial=prepared[name]['rotation']
                    for attempt in range(config['max_start_trials']):
                        key=(tuple(position),tuple(initial))
                        if key in seen:break
                        seen.add(key)
                        trace=execute(position,initial,history)
                        pose=trace['observations'][-1]['pose']
                        matched=exact_pose_equal(pose,target) and raw_window(trace,len(history))==target_window and compiler.complete(trace)
                        item=dict(attempt=attempt,initial_position=position,initial_rotation=initial,
                            final_pose=pose,pose_delta=pose_delta(pose,target),raw_equal=raw_window(trace,len(history))==target_window,
                            collisions=trace['collisions'],matched=matched)
                        attempts.append(item)
                        c.append(folder/(name+'_ATTEMPTS.jsonl'),item)
                        if matched:
                            c.write(folder/(name+'_PREFIX.json'),trace,True)
                            starts[name]=dict(position=position,rotation=initial,attempt=attempt)
                            if name in family.get('neutral_span',{}):
                                controls[name]=neutral_span(compiler,trace,family['neutral_span'][name])
                            if name in family.get('balancing_filler_spans',{}):
                                controls[name+':filler']=neutral_span(compiler,trace,family['balancing_filler_spans'][name])
                            break
                        if trace['collisions']:break
                        if prepared:break
                        # Change only the next episode's initial condition. Every trial
                        # replays the complete history through ordinary simulator steps.
                        position=[float(backend.np.float32(p+t-v)) for p,t,v in zip(position,target['position'],pose['position'])]
                        q=backend.np.quaternion(*target['rotation'])/backend.np.quaternion(*pose['rotation'])*backend.np.quaternion(*initial)
                        initial=(backend.quaternion.as_float_array(q)/abs(q)).tolist()
                    c.write(folder/(name+'_CALIBRATION.json'),dict(attempts=attempts,matched=name in starts),True)
                grid={}
                if len(starts)==len(candidate['histories']) and all(x['pass_control'] for x in controls.values()):
                    for name,history in candidate['histories'].items():
                        start=starts[name]
                        for suffix_name,suffix in candidate['continuations'].items():
                            trace=execute(start['position'],start['rotation'],history+suffix)
                            cutoff=len(history)
                            assert exact_pose_equal(trace['observations'][cutoff]['pose'],target)
                            assert raw_window(trace,cutoff)==target_window
                            key=name+'__'+suffix_name
                            c.write(folder/(key+'.json'),trace,True)
                            grid[key]=dict(labels={task:compiler.evaluate(trace,task) for task in compiler.tasks},
                                           terminal_only=terminal_only(compiler,trace),
                                           collisions=trace['collisions'],decisions=len(trace['actions']))
                result=dict(family_id=family['family_id'],house=family['house'],starts=starts,grid=grid,
                    exact_raw_join=len(starts)==len(candidate['histories']),training_admission=False,
                    neutral_controls=controls,
                    pose_comparison='Exact positions and rotation coefficients up to quaternion sign only; no tolerance, state changes, or RGB transformations.',
                    reason='Initial-state interface only; true neutral detour and independent data remain required.',
                    backend_counts=backend.counts)
                if controls:
                    result['relations_pass']=bool(grid) and all(
                        grid[h+'__'+q]['labels']==grid[h+'_N__'+q]['labels']
                        and grid[h+'__'+q]['terminal_only']==grid[h+'_N__'+q]['terminal_only']=='pass'
                        for h in ('H_A','H_B') for q in candidate['continuations']) and (
                        grid['H_A__C0']['labels']=={'task_A':'pass','task_B':'fail'} and
                        grid['H_B__C0']['labels']=={'task_A':'fail','task_B':'pass'})
                c.write(folder/'CERTIFICATE.json',result,True);results.append(result)
                backend.close();backend=None
                if result.get('relations_pass') and config.get('stop_after_first_valid_control'):
                    break
            store._check_root()
            audit=store.full_audit();assert audit['audit_pass']
            c.write(run/'CONTENT_AUDIT.json',audit,True)
            c.write(run/'TIMESTAMP_CHECKS.json',store.timestamp_checks,True)
        c.write(run/'RESULT.json',dict(status='INITIAL_STATE_CALIBRATION_MEASURED',families=results,
            unattempted_families=[f['family_id'] for f in config['families'] if f['family_id'] not in {r['family_id'] for r in results}],
            actual_decisions=decisions,model_loads=0,optimizer_updates=0,training_admission=False),True)
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
