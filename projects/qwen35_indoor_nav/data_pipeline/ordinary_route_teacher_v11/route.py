"""Privileged offline teacher target cursor; never passed to the policy."""
import math
REACH=.35
def advance(targets,cursor,distance):
    assert targets and type(cursor) is int and 0<=cursor<=len(targets)
    tested=[]
    while cursor<len(targets):
        d=float(distance(targets[cursor]))
        assert not math.isnan(d) and d>=0
        tested.append(dict(index=cursor,distance=d if math.isfinite(d) else None))
        if not math.isfinite(d) or d>REACH:break
        cursor+=1
    selected=min(cursor,len(targets)-1)
    if cursor==len(targets):
        d=float(distance(targets[-1]))
        assert not math.isnan(d) and d>=0
    result=dict(cursor_after=cursor,selected_index=selected,tested=tested,
        selected_distance=d if math.isfinite(d) else None,
        route_complete=cursor==len(targets))
    return result
def stop_at(route,final_distance):
    return route['route_complete'] and math.isfinite(final_distance) and final_distance<=REACH
