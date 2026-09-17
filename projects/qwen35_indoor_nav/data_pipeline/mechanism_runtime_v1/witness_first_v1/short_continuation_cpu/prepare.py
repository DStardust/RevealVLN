"""CPU geometric proposals retaining a witnessed consecutive-frame edge.

No replay, pixel synthesis, task/checker edits, or history changes. Exact original
protected poses are replay targets, not claims that a compressed path reaches them.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler
from factory import compress


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ideal(actions):
    x=z=0.0
    yaw=0
    for action in actions:
        if action=='L':yaw=(yaw+1)%24
        elif action=='R':yaw=(yaw-1)%24
        elif action=='F':
            x-=.25*math.sin(yaw*math.pi/12)
            z-=.25*math.cos(yaw*math.pi/12)
        else:raise ValueError('MOTION_ONLY')
    return x,z,yaw


def equivalent(a,b):
    x,z,y=ideal(a);u,v,w=ideal(b)
    return y==w and math.hypot(x-u,z-v)<1e-12


def protected_compress(actions,t):
    if type(t) is not int or not 1<=t<=len(actions):raise ValueError('PROTECTED_PAIR_STEP')
    before=compress(actions[:t-1])
    edge=list(actions[t-1:t])
    after=compress(actions[t:])
    result=before+edge+after
    if not equivalent(actions[:t-1],before) or not equivalent(actions[:t],before+edge) or not equivalent(actions,result):
        raise ValueError('IDEAL_GEOMETRY_DIFFERENT')
    return result,[len(before),len(before)+1]


def proposals(actions,trace,eligible,limit=3):
    if not 1<=limit<=3:raise ValueError('BOUNDED_LIMIT')
    if trace['actions'][:len(actions)]!=actions or not trace['complete'] or trace['collisions']:
        raise ValueError('SOURCE_TRACE_NOT_VALID_COMPONENT_PREFIX')
    result={}
    for t in range(1,len(actions)+1):
        previous,current=trace['observations'][t-1:t+1]
        witnesses=[i for i in eligible if previous['pixels'].get(str(i),0)>=256 and current['pixels'].get(str(i),0)>=256]
        if not witnesses or previous['evidence_complete'] is not True or current['evidence_complete'] is not True:continue
        candidate,pair=protected_compress(actions,t)
        pixels=max(min(previous['pixels'].get(str(i),0),current['pixels'].get(str(i),0)) for i in witnesses)
        row={'actions':candidate,'motion_actions':len(candidate),'protected_source_frames':[t-1,t],
             'protected_new_frames':pair,'protected_edge_action':actions[t-1],
             'protected_source_poses':[previous['pose'],current['pose']],
             'protected_source_rgb_hashes':[previous['rgb_hash'],current['rgb_hash']],
             'protected_source_semantic_hashes':[previous['semantic_hash'],current['semantic_hash']],
             'protected_witness_ids':witnesses,'protected_min_pixels':pixels,
             'proposal_kind':'compress_turn_runs_before_and_after_one_untouched_observation_edge',
             'ideal_geometry_equal':True,'actual_protected_pose_match':None,
             'actual_event_preserved':None,'actual_endpoint_closure':None,
             'actual_query_length':None,'physical_replay_pass':None}
        key=tuple(candidate)
        if key not in result or (-pixels,t)<(-result[key]['protected_min_pixels'],result[key]['protected_source_frames'][1]):
            result[key]=row
    return sorted(result.values(),key=lambda r:(r['motion_actions'],-r['protected_min_pixels'],r['protected_source_frames']))[:limit]


def prepare():
    source=WF/'revisit_v1/run_v1'
    cfg_path=source/'EXECUTION_CONFIG.json'
    trace_path=source/'bundles/WF_REVISIT_004/traces/000001.json'
    cfg=json.loads(cfg_path.read_text())
    row=cfg['candidates'][0]
    trace=json.loads(trace_path.read_text())
    a=row['components']['a']
    candidates=proposals(a,trace,row['expected_eligible']['anchor_A'])
    if not candidates:raise ValueError('NO_PROTECTED_WITNESS_CANDIDATE')
    for i,candidate in enumerate(candidates):
        candidate['candidate_id']='WF_SHORT_A_%02d'%i
        candidate['continuation_C_A']=candidate['actions']+row['components']['terminal']+['S']
        candidate['continuation_actions_including_stop']=len(candidate['continuation_C_A'])
    original_matrix_trace=source/'bundles/WF_REVISIT_004/traces/000007.json'
    full=json.loads(original_matrix_trace.read_text())
    roles={k:(v['mpcat40'],v['room']) for k,v in row['roles'].items()}
    compiler=Compiler(roles,row['tasks'],row['expected_eligible'])
    cutoff=len(full['actions'])-len(a)-len(row['components']['terminal'])-1
    original_query=compiler.query_from_trace(compiler.slice_continuation(full,cutoff))
    sources=[cfg_path,trace_path,original_matrix_trace,RUNTIME/'core_bridge.py',
             RUNTIME.parent/'mechanism_factory_v2/factory.py',
             RUNTIME.parent/'mechanism_factory_v2/compiler.py',
             WF/'revisit_v1/method.py',WF/'assembly_v1/method.py',HERE/'prepare.py']
    result={'node':'Q35N_SHORT_PROTECTED_CONTINUATION_CPU_V1',
        'source_hashes':{str(p):sha(p) for p in sources},
        'candidate_count':len(candidates),'candidates':candidates,
        'original_A_component_actions':len(a),'original_C_A_actions':len(a)+len(row['components']['terminal'])+1,
        'original_observed_query_sequence_length':len(original_query['sequence']),
        'query_sequence_limit_unchanged':160,'original_checker_unchanged':True,
        'original_history_components_unchanged':row['components'],
        'continuation_only_override_key':'continuation_a; do not overwrite components.a used by histories',
        'history_motion_actions_before_public_tail':240,
        'runtime_executed':False,'training_admission':False,'scientific_pass':False,
        'required_replay_checks':['protected pair actual pose and witness preserved','zero collisions',
           'original 1e-5 common-state closure','same original checker','query length <=160',
           'complete 3x3x2 and three seed replay','all original causal and shortcut checks']}
    with (HERE/'CANDIDATES.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['candidate_count','original_A_component_actions','original_observed_query_sequence_length']}))
    print(json.dumps([(c['candidate_id'],c['motion_actions'],c['protected_source_frames'],c['protected_new_frames']) for c in candidates]))
    return result


if __name__=='__main__':prepare()
