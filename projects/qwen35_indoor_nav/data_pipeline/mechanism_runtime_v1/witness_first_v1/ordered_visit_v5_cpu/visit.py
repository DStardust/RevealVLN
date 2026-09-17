"""Offline-only V5 state prototype. Not a replacement for the sealed compiler."""
import math

RADIUS_M=1.0
PIXELS=256

def _valid(frame,roles):
    if not isinstance(frame,dict) or frame.get('evidence_verified') is not True:return False
    for role in roles:
        item=frame.get(role)
        if not isinstance(item,dict):return False
        distance=item.get('zone_geodesic_m');pixels=item.get('visible_instance_pixels')
        if type(distance) not in (int,float) or not math.isfinite(distance) or distance<0:return False
        if not isinstance(pixels,dict) or any(type(v) is not int or v<0 for v in pixels.values()):return False
        if item.get('zone_certified') is not True:return False
    return True

def _near_pair(prev,now,role,eligible):
    if prev[role]['zone_geodesic_m']>RADIUS_M or now[role]['zone_geodesic_m']>RADIUS_M:return False
    return any(prev[role]['visible_instance_pixels'].get(i,0)>=PIXELS and
               now[role]['visible_instance_pixels'].get(i,0)>=PIXELS for i in eligible[role])

def evaluate(trace,eligible):
    roles=('A','B')
    if (set(eligible)!=set(roles) or any(not eligible[r] for r in roles) or
        trace.get('zones_disjoint_verified') is not True):return dict(y=None,reason='UNVERIFIED_TASK_ASSETS')
    observations=trace.get('observations',[]);actions=trace.get('actions',[])
    if (trace.get('complete') is not True or trace.get('collisions')!=0 or
        not actions or len(observations)!=len(actions)+1 or
        any(a not in ('F','L','R','S') for a in actions) or 'S' in actions[:-1] or
        not all(_valid(o,roles) for o in observations)):
        return dict(y=None,reason='UNVERIFIED_OR_INCOMPLETE_TRACE')
    arrival_a=arrival_b=None;states=[]
    for t,now in enumerate(observations):
        a=t>0 and _near_pair(observations[t-1],now,'A',eligible)
        b=t>0 and _near_pair(observations[t-1],now,'B',eligible)
        if a and b:return dict(y=None,reason='OVERLAPPING_ACCEPTANCE_REGIONS')
        if a and arrival_a is None:arrival_a=t
        # Both B evidence frames must occur after completing A.
        if b and arrival_a is not None and t-1>arrival_a and arrival_b is None:arrival_b=t
        states.append(dict(step=t,completed_A=arrival_a is not None,
                           completed_B_in_order=arrival_b is not None,currently_at_B=bool(b)))
    passed=actions[-1]=='S' and arrival_b is not None and states[-1]['currently_at_B']
    return dict(y=int(passed),reason='PASS' if passed else 'LEGAL_TASK_FAILURE',
                arrival_A=arrival_a,arrival_B=arrival_b,strong_state_supervision=states,
                policy_input_allowed=False,training_admission=False)
