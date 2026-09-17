"""3D non-overlapping hub coverage, with no simulator or optional imports."""
import math

def point(value):
    if len(value)!=3 or not all(type(x) in (int,float) and math.isfinite(x) for x in value):raise ValueError('FINITE_3D_POINT_REQUIRED')
    return list(value)
def choose(positions,geometry,excluded,limit=4):
    if limit!=4:raise ValueError('FIXED_FOUR_HUB_PILOT')
    excluded=[point(p) for p in excluded];seen=set();rows=[];ledger=[]
    geom={tuple(r['position']):r for r in geometry}
    for i,value in enumerate(positions):
        p=point(value);key=tuple(p);item={'source_index':i,'position':p,'status':None};ledger.append(item)
        if key in seen:item['status']='DUPLICATE_SOURCE_POSITION';continue
        seen.add(key)
        if any(math.dist(p,old)<1 for old in excluded):item['status']='WITHIN_1M_PREVIOUS_OR_DECLARED_HUB';continue
        record=geom.get(key)
        if record is None:item['status']='PRIOR_REACHABILITY_UNTESTED';continue
        item['prior_geometry']=record
        if record['status']!='GEOMETRY_CHECKED_NOT_VISIBILITY_CERTIFIED' or record.get('snapped') is None or math.dist(p,record['snapped'])>1e-5:
            item['status']='PRIOR_COORDINATE_NOT_EXACT_NAVMESH';continue
        if record['reachable_groups']<=0:item['status']='NO_PRIOR_REACHABLE_ROLE_GROUP';continue
        rows.append(item)
    rows.sort(key=lambda r:(-r['prior_geometry']['reachable_groups'],r['position']))
    selected=[]
    for item in rows:
        if any(math.dist(item['position'],r['position'])<1 for r in selected):item['status']='WITHIN_1M_NEW_SELECTED_HUB'
        elif len(selected)>=limit:item['status']='NOT_SELECTED_FOUR_HUB_BUDGET'
        else:item['status']='PROSPECTIVE_SELECTED_REQUIRES_LIVE_GEOMETRY_RECHECK';selected.append(item)
    return selected,ledger
def runtime_select(original,backend,role_rows,positions,budget,cfg,target_function):
    assert len(cfg['candidates'])==1 and cfg['candidates'][0]['house_id']==cfg['new_hub_plan']['house_id']
    plan=cfg['new_hub_plan'];expected=plan['selected_positions'];excluded=plan['excluded_positions']
    assert positions==expected and len(expected)==4 and len({tuple(point(x)) for x in expected})==4
    for i,p in enumerate(expected):
        assert all(math.dist(p,q)>=1 for q in excluded+expected[:i]),'HUB_OVERLAP'
    hubs,geometry=original(backend,role_rows,positions,budget,target_function)
    assert len(hubs)<=4
    for h in hubs:assert h['position'] in expected and h['yaw_bin']==0
    geometry.update(new_hub_selection=True,previous_hubs_excluded=excluded,
        frozen_selected_positions=expected,all_four_runtime_geometry_retained=len(hubs)==4,
        selection_changes_navigation_algorithm=False,not_generalization_evaluation=True)
    return hubs,geometry
