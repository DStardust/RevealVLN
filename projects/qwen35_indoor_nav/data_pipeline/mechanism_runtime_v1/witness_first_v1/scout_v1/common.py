"""Read-only adapters and deterministic geometry proposal logic for scout V1."""
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
ROOT=RUNTIME.parents[3]
LINE=RUNTIME.parents[1]
sys.path[:0]=[str(RUNTIME),str(RUNTIME/'feedback_generation_v1'),str(HERE.parent/'budget_and_trace')]
from core_bridge import Compiler,compiler,digest,factory,BudgetExceeded
from adapters import Journal,durable_budget,PartialTraceRunner
from feedback import candidate_targets,FeedbackRunner
from habitat_backend import HabitatBackend

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

bank=load('scout_frozen_bank',HERE.parent/'bank_cpu/bank.py')

def sha(path):
    import hashlib
    path=Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)

def store_class():
    module=load('scout_explicit_new_store',RUNTIME/'feedback_generation_v1/store.py')
    module.FEEDBACK_ROOT=HERE
    return module.TrackedContentStore

class ProbeBackend:
    """No mutation of the actual backend's original four permission roles."""
    def __init__(self,backend,ids):self._backend=backend;self.eligible={'probe':list(ids)}
    def __getattr__(self,name):return getattr(self._backend,name)

def registry(objects):
    groups=bank.compat.role_groups(objects)
    return [dict(signature=list(sig),eligible_ids=ids) for sig,ids in groups.items()]

def select_hubs(backend,role_rows,positions,budget,target_function=candidate_targets):
    rows=[];checks=[]
    for p in sorted({tuple(p) for p in positions}):
        budget.check_time();p=list(p);snapped=backend.snap_position(p)
        if snapped is None or math.dist(p,snapped)>1e-5:
            checks.append(dict(position=p,status='SOURCE_COORDINATE_NOT_EXACT_NAVMESH',snapped=snapped));continue
        reachable=[]
        for role in role_rows:
            budget.check_time()
            targets=target_function(ProbeBackend(backend,role['eligible_ids']),p,'probe',limit=2)
            if targets:
                assert len(targets)<=2 and all(0<=t['distance']<=8 for t in targets)
                reachable.append(dict(role=role,targets=targets,distance=targets[0]['distance']))
        reachable.sort(key=lambda r:(r['distance'],r['role']['signature']))
        checks.append(dict(position=p,status='GEOMETRY_CHECKED_NOT_VISIBILITY_CERTIFIED',
            reachable_groups=len(reachable),snapped=snapped))
        if reachable:
            rows.append(dict(position=p,yaw_bin=0,reachable_groups=len(reachable),
                nearest_target_m=reachable[0]['distance'],sum_nearest_distances_m=sum(r['distance'] for r in reachable),
                groups=reachable[:12],public_tail='LRLRLRLR'))
    rows.sort(key=lambda r:(-r['reachable_groups'],r['nearest_target_m'],r['sum_nearest_distances_m'],r['position']))
    selected=[]
    for row in rows:
        if all(math.dist(row['position'],old['position'])>=1 for old in selected):selected.append(row)
        if len(selected)==2:break
    return selected,dict(source_positions=len(positions),geometry_checks=checks,
        viable_geometry_positions=len(rows),selected_hubs=len(selected),physical_or_visibility_certified=False)

class ScoutFeedbackRunner(FeedbackRunner):
    """Original feedback operator with fresh partial-capture state and error ledger."""
    def navigate(self,*args,**kwargs):
        self.base.capture.clear();self.base._validated=[]
        try:return super().navigate(*args,**kwargs)
        except BaseException as error:
            self.emit('feedback_exception',dict(error=repr(error),
                confirmed_actions=len(self.base.capture.confirmed),
                unknown_attempts=self.base.capture.unknown_attempts,
                validated_observations=len(self.base._validated),training_admission=False))
            raise

def compact_actions(outbound):
    if any(a not in ('F','L','R') for a in outbound):raise ValueError('MOTION_ONLY')
    # Construct the compact inverse directly: never build or execute expanded raw inverse.
    inverse=[]
    for action in reversed(outbound):
        fragment=['L']*12+['F']+['R']*12 if action=='F' else ['R' if action=='L' else 'L']
        inverse=factory.compress(inverse+fragment)
    return list(outbound)+inverse

def runtime_config():
    lock=json.loads((HERE/'INPUT_LOCK.json').read_text())
    for name,expected in lock.items():assert sha(ROOT/name)==expected,name
    approval=json.loads((HERE/'MAIN_AGENT_APPROVAL.json').read_text())
    assert approval==dict(approved=True,gpu=1,input_lock_sha256=sha(HERE/'INPUT_LOCK.json'))
    cfg=json.loads((HERE/'PREPARED_CONFIG.json').read_text())
    assert cfg['runtime_allowed'] is False and cfg['training_allowed'] is False
    cfg.update(runtime_allowed=True,executable=True,main_agent_approval_sha256=sha(HERE/'MAIN_AGENT_APPROVAL.json'))
    return cfg
