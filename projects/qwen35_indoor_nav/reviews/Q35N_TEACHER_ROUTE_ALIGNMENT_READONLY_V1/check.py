"""Read-only goal-teacher/ordered-route diagnostic; no relabel, no simulator."""
import collections,hashlib,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
TRAIN=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ep=read(FIT/'EPISODES_PRIVILEGED.json')
    eligible={x['advice_record_id'] for x in read(TRAIN/'TRAINING_ROWS.json')[37114:]}
    traces={x['index']:x for x in (read(p) for p in (FIT/'run_001/lanes').glob('lane_*/episode_*.json'))}
    assert len(traces)==64 and len(eligible)==4983
    first={}
    for r in rows(DATA/'SUPERVISION_ONLY.jsonl'):
        if r['record_id'] in eligible and r['record_id'] not in first:first[r['record_id']]=r
    assert set(first)==eligible
    progress={}
    for i,e in enumerate(ep):
        path=e['reference_path'];cursor=0
        for t,pos in enumerate(traces[i]['positions'][:-1]):
            while cursor<len(path) and math.dist(pos,path[cursor])<=.35:cursor+=1
            distances=[math.dist(pos,wp) for wp in path]
            nearest=min(range(len(path)),key=distances.__getitem__)
            tail=sum(math.dist(a,b) for a,b in zip(path[nearest:],path[nearest+1:]))
            progress[i,t]=dict(euclidean_ordered_cursor=cursor,reference_waypoints=len(path),
                nearest_reference_index=nearest,nearest_distance=distances[nearest],nearest_remaining_reference_polyline_m=tail,
                never_reached_ordered_reference_endpoint=cursor<len(path),original_route_success=bool(traces[i]['success']))
    counts=collections.Counter();examples=[]
    for rid,r in first.items():
        x=progress[r['source_index'],r['decision_step']];counts['eligible_unique_inputs']+=1
        if r['target']==3:
            counts['teacher_STOP']+=1
            counts['STOP_original_success_route' if x['original_route_success'] else 'STOP_original_failed_route']+=1
            if x['never_reached_ordered_reference_endpoint']:counts['STOP_before_ordered_reference_endpoint_euclidean_necessary_condition']+=1
            if r['original_action']!='STOP':counts['STOP_disagrees_with_original_continuation']+=1
            examples.append(dict(record_id=rid,source_index=r['source_index'],episode_id=r['episode_id'],
                instruction=ep[r['source_index']]['instruction']['instruction_text'],decision_step=r['decision_step'],
                original_action=r['original_action'],distance_to_goal=r['distance_to_goal'],**x))
    examples.sort(key=lambda x:(-x['nearest_remaining_reference_polyline_m'],x['source_index'],x['decision_step']))
    result=dict(status='COMPLETE_DIAGNOSTIC_NOT_LABEL_VALIDITY_PASS',unix=time.time(),counts=dict(counts),
        stop_examples_by_remaining_polyline=examples[:20],
        limitations=['Euclidean waypoint reach is only a necessary condition for original geodesic .35m reach, not new replay certification',
          'Original benchmark accepts explicit STOP at final goal distance<3m; early relative to reference is not automatically an invalid benchmark target',
          'Nearest waypoint and remaining polyline are descriptive and NOT new expert targets',
          'This cannot establish that supervision mismatch caused the observed navigation losses'],
        official_code_reference='https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/sensors.py',
        official_source_observation='ShortestPathSensor queries episode.goals[0].position; endpoint teacher is known ordinary engineering, not itself an implementation error',
        policy_inputs_modified=0,labels_modified=0,gpu_actions=0,training_updates=0,
        sources={str(p):sha(p) for p in [FIT/'EPISODES_PRIVILEGED.json',TRAIN/'TRAINING_ROWS.json',DATA/'SUPERVISION_ONLY.jsonl',Path(__file__)]})
    with (HERE/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(counts=dict(counts),top_examples=examples[:3]),ensure_ascii=False))
if __name__=='__main__':main()
