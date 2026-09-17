"""Pure causal-input and physical-advice contracts; no simulator/model access."""
import hashlib,json,math
ACTIONS=['move_forward','turn_left','turn_right','STOP']
def stop_at(distance):
    assert math.isfinite(distance) and distance>=0
    return distance<3.0
def payload(instruction,images,executed):
    assert isinstance(instruction,str) and instruction
    assert 1<=len(images)<=2 and all(isinstance(x,str) and len(x)==64 for x in images)
    assert len(executed)<=8 and all(x in ACTIONS[:3] for x in executed)
    return dict(instruction=instruction,rgb_sha256=list(images),executed_actions=list(executed))
def key(value):
    assert set(value)=={'instruction','rgb_sha256','executed_actions'}
    payload(value['instruction'],value['rgb_sha256'],value['executed_actions'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def displacement(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def angle(a,b):
    na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b))
    assert na>0 and nb>0
    dot=abs(sum(x*y for x,y in zip(a,b))/(na*nb))
    return math.degrees(2*math.acos(max(-1.,min(1.,dot))))
def movement_valid(action,before,after,collided,before_distance,after_distance):
    if collided:return False
    delta=displacement(before['position'],after['position']);turn=angle(before['rotation'],after['rotation'])
    if action=='move_forward':return .015<delta<=.2501 and turn<.001 and after_distance<before_distance-1e-4
    if action in ('turn_left','turn_right'):return delta<1e-5 and abs(turn-15.)<.001
    return False
def admissible_group(records):
    targets={x['target'] for x in records}
    return len(targets)==1 and None not in targets
