"""Prospective fixed [1,2,3] source recipe. CPU only; never reads runtime outcomes."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
EXEC=HERE.parent
WF=EXEC.parent
RUNTIME=WF.parent
ROOT=RUNTIME.parents[3]
SOURCE=WF/'winding_balance_v1/next_source_v1/snapshot_v2'
INDICES=(1,2,3)
EXPECTED=(('2n8kARJN3HM',1),('5LpN3gDmAk7',0),('29hnd4uzFmX',0))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


lang=load('batch02_frozen_verbalizer',WF/'language_realization_v1/verbalizer.py')
method=load('batch02_frozen_winding',WF/'winding_balance_v1/method.py')
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler


def sha(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False,ensure_ascii=False)


def select(draft):
    assert not draft['runtime_allowed'] and not draft['executable'] and not draft['training_allowed']
    rows=[copy.deepcopy(draft['candidates'][i]) for i in INDICES]
    assert tuple((r['house_id'],r['hub_index']) for r in rows)==EXPECTED
    assert all(r['split']=='FIT' and not r['runtime_allowed'] and not r['executable'] for r in rows)
    return rows


def independent(row,other):
    if row['house_id']!=other['house_id']:
        return dict(same_house=False,distance_m=None,at_least_1m=True)
    distance=math.dist(row['configuration']['u_position'],other['configuration']['u_position'])
    assert distance>=1.0,'PRESELECTED_HUB_LT_1M'
    return dict(same_house=True,distance_m=distance,at_least_1m=True)


def surface_revision(row):
    old=copy.deepcopy(row);revision=lang.propose_revision(old)
    revised=copy.deepcopy(old);revised['tasks']=revision['proposed_tasks']
    # The complete recipe body, apart from task wording, must remain byte-equivalent
    # under canonical JSON before adding explicit version/provenance annotations.
    before=copy.deepcopy(old);after=copy.deepcopy(revised)
    for task in before['tasks'].values():task.pop('instruction')
    for task in after['tasks'].values():task.pop('instruction')
    assert before==after,'NON_LANGUAGE_RECIPE_MUTATION'
    revised['candidate_id']=old['candidate_id']+'_B02_LR1'
    revised['language_version']=lang.VERSION
    revised['component_provenance']['language_revision']=revision
    revised['component_provenance']['prospective_batch02_selection']=dict(
        source_snapshot=str(SOURCE),source_candidate_id=old['candidate_id'],
        fixed_indices=list(INDICES),selection_is_not_runtime_outcome_dependent=True)
    return revised,revision


class NoRuntime:
    def __getattr__(self,name):
        raise AssertionError('CPU_CONSTRUCTOR_ATTEMPTED_RUNTIME: '+name)


def validate_constructor(old,row):
    roles={k:(r['mpcat40'],r['room']) for k,r in row['roles'].items()}
    compiler=Compiler(roles,row['tasks'],row['expected_eligible'])
    previous=Compiler(roles,old['tasks'],row['expected_eligible'])
    for task in row['tasks']:assert compiler.task_program(task)==previous.task_program(task)
    factory=method.BalancedFactory(NoRuntime(),compiler,NoRuntime(),lambda *args:None,
        'task_A','task_B',context={'house_id':row['house_id'],'asset_config':row['assets']},
        components=row['components'],balance=row['balance'])
    plan=factory.padding_plan
    assert plan['action_counts']==row['expected_history_action_counts']
    assert plan['history_actions']==row['history_actions_before_public_tail']
    assert plan['pads']==row['winding_padding_plan']
    assert plan['required_neutral_spin_directions']==row['required_neutral_spin_directions']
    counts={name:{a:actions.count(a) for a in ('F','L','R')} for name,actions in plan['histories'].items()}
    assert all(c==plan['action_counts'] for c in counts.values())
    continuation=factory.continuations(None,None)
    lengths={k:len(v) for k,v in continuation.items()}
    assert max(lengths.values())==row['max_continuation_actions']<=160
    tail=row['configuration']['public_tail']
    assert tail=='LRLRLRLR' and plan['history_actions']+len(tail)<=512
    return dict(candidate_id=row['candidate_id'],house_id=row['house_id'],hub_index=row['hub_index'],
        cpu_constructor_pass=True,backend_calls=0,task_predicates_unchanged=True,
        history_actions_before_public_tail=plan['history_actions'],public_tail_actions=len(tail),
        history_actions_including_public_tail=plan['history_actions']+len(tail),
        exact_history_FLR_counts_before_tail=plan['action_counts'],continuation_actions=lengths,
        source_observation_query_estimates=row['source_observation_query_estimates'],
        query_estimates_not_new_runtime_observations=True,
        required_neutral_spin_directions=plan['required_neutral_spin_directions'],
        physical_replay_certified=False,training_admission=False)


def main():
    assert not (HERE/'composite').exists(),'FRESH_OUTPUT_REQUIRED'
    lock={}
    def merge(path):
        for name,expected in json.loads(path.read_text()).items():
            p=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(p)==expected,str(p)
            key=str(p.resolve());assert key not in lock or lock[key]==expected
            lock[key]=expected
        lock[str(path.resolve())]=sha(path)
    merge(SOURCE/'SOURCE_LOCK.json')
    merge(WF/'language_realization_v1/INPUT_LOCK.json')
    draft=json.loads((SOURCE/'CONFIG_DRAFT.json').read_text());rows=select(draft)
    lock[str(SOURCE/'CONFIG_DRAFT.json')]=sha(SOURCE/'CONFIG_DRAFT.json')
    old=[]
    for batch in ('batch_00','batch_01r1'):
        p=EXEC/f'{batch}/run_v1/EXECUTION_CONFIG.json'
        cfg=json.loads(p.read_text());lock[str(p)]=sha(p)
        old.extend((batch,r) for r in cfg['candidates'])
    comparisons=[];revisions=[];validated=[];new=[]
    for index,row in enumerate(rows):
        for batch,other in old+[('prospective_batch02',r) for r in rows[:index]]:
            comparisons.append(dict(candidate_id=row['candidate_id'],reference_batch=batch,
                reference_candidate_id=other['candidate_id'],**independent(row,other)))
        realized,revision=surface_revision(row)
        revisions.append(revision);validated.append(validate_constructor(row,realized));new.append(realized)
        for p,h in row['assets'].items():assert sha(p)==h,p
    cfg=dict(node='Q35N_BATCH02_PROSPECTIVE_FIXED_HUB_COMPOSITE',factory_variant='winding_v1',
        runtime_allowed=False,executable=False,training_allowed=False,candidates=new,
        source_selection=[dict(snapshot=str(SOURCE),indices=list(INDICES))],
        selection_rule='main-agent prospective fixed snapshot_v2 indices 1,2,3; preserve all attempted batches and failures',
        selection_uses_runtime_outcomes=False,language_version=lang.VERSION,
        source_roles_actions_eligibility_and_task_predicates_unchanged=True,
        query_sequence_limit=160,seed_replays=[1109,2209,3309],requires_main_agent_admission=True,
        resource_budget=None,gpu_device=None)
    for p in list(HERE.glob('*.py'))+[HERE/'README_ZH.md',WF/'language_realization_v1/SCHEMA.json']:
        lock[str(p)]=sha(p)
    for p,h in lock.items():assert sha(p)==h,p
    out=HERE/'composite';out.mkdir()
    for name,value in [('CONFIG_DRAFT.json',cfg),('SOURCE_LOCK.json',lock),
        ('LANGUAGE_REVISIONS.json',revisions),('HUB_DISTANCE_CHECKS.json',comparisons),
        ('CPU_CONSTRUCTOR_ACCEPTANCE.json',validated)]:save(out/name,value)
    summary=dict(status='CPU_BATCH02_SOURCE_READY_NOT_EXECUTABLE',selected_indices=list(INDICES),
        selected_houses=[r['house_id'] for r in new],selected_hubs=[r['hub_index'] for r in new],
        validated_constructors=len(validated),source_hashes_verified=len(lock),
        changed_instructions=sum(len(r['changes']) for r in revisions),
        prior_candidate_comparisons=18,within_new_batch_comparisons=3,
        no_same_house_hub_below_1m=True,gpu_operations=0,simulator_runs=0,
        runtime_allowed=False,executable=False,scientific_pass=False,validation=validated)
    save(out/'result.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
