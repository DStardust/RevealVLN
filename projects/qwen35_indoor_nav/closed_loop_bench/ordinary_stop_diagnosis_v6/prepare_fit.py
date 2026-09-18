"""Six real FIT routes with offline geodesic STOP labels; no renderer or training.

Original action labels stay intact. This is a small data-interface check, not a
new navigation result, an independent test set, or a trained STOP policy.
"""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
ROOT=LINE.parents[1]
SNAPSHOT=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
DATASET=ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_payload(row):
    value=row['policy']
    if set(value)!={'instruction','rgb_paths','executed_actions'}:
        raise ValueError('PRIVILEGED_POLICY_FIELD')
    if not 1<=len(value['rgb_paths'])<=2 or len(value['executed_actions'])>8 or 'STOP' in value['executed_actions']:
        raise ValueError('CAUSAL_WINDOW_CONTRACT')
    return value


def stop_target(distance):
    return dict(can_stop=int(distance<3.) if math.isfinite(distance) else None,
                loss_mask=int(math.isfinite(distance)))


def main():
    # Habitat PathFinder loads a navmesh without creating a Simulator/GL context.
    import habitat_sim
    spec=importlib.util.spec_from_file_location('v6_official_distance',HERE.parent/'ordinary_cycle_pair_gpu1_v3/official_distance.py')
    distance_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(distance_module)
    rows=[json.loads(x) for x in (SNAPSHOT/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    split=read(SNAPSHOT/'SPLIT.json')
    candidates=sorted((r for r in rows if r['source']=='R2R' and r['split']=='FIT'),key=lambda r:r['record_id'])
    houses=sorted({r['scene_group'] for r in candidates})[:3]
    assert set(houses).issubset(set(split['FIT']))
    assert set(houses).isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    selected=[]
    for house in houses:
        seen=set()
        for row in candidates:
            key=row['physical_source_route_sha256']
            if row['scene_group']==house and key not in seen:
                selected.append(row);seen.add(key)
                if len(seen)==2:break
        assert len(seen)==2
    out=HERE/'fit_sample_001';out.mkdir()
    episodes={str(e['episode_id']):e for e in json.load(gzip.open(DATASET))['episodes']}
    counts={'decisions':0,'positive':0,'negative':0,'unknown':0,'positive_with_motion_teacher':0}
    summaries=[]
    with (out/'DECISIONS.jsonl').open('x') as stream:
        for house in houses:
            navmesh=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.navmesh'
            finder=habitat_sim.PathFinder();assert finder.load_nav_mesh(str(navmesh))
            for row in (r for r in selected if r['scene_group']==house):
                root=ROOT/row['sourceRoot']
                policy_path=root/row['policy_file'];supervision_path=root/row['supervision_file']
                assert sha(policy_path)==row['policy_sha256'] and sha(supervision_path)==row['supervision_sha256']
                policy,supervision=read(policy_path),read(supervision_path)
                certificate=read(supervision_path.parent/'replay_certificate.json')
                assert certificate['exact_all_frame_pose_rgb_semantic_match']
                episode=episodes[policy_path.stem.replace('policy_','',1)]
                source_episode=episodes[str(supervision['source_episode_id'])]
                assert episode['goals']==source_episode['goals']
                assert Path(episode['scene_id']).stem==house
                assert policy['instruction']==episode['instruction']['instruction_text']
                actions=supervision['actions'];frames=supervision['frames']
                assert len(actions)==len(frames)==len(policy['rgb_sequence'])==row['decisions']
                assert actions[-1]=='STOP' and 'STOP' not in actions[:-1]
                goals=[episode['goals'][0]['position']]
                cache=SimpleNamespace(_shortest_path_cache=None)
                route_positive=0;last_distance=None
                partition='head_fit' if house in houses[:2] else 'head_check'
                for t,(action,frame) in enumerate(zip(actions,frames)):
                    distance=distance_module.measure(finder,frame['position'],goals,cache)
                    target=stop_target(distance)
                    rgb=[str((root/x).relative_to(ROOT)) for x in policy['rgb_sequence'][max(0,t-1):t+1]]
                    assert all((ROOT/x).is_file() for x in rgb)
                    sample=dict(sample_id=row['record_id']+':'+str(t),partition=partition,
                        policy=dict(instruction=policy['instruction'],rgb_paths=rgb,executed_actions=actions[max(0,t-8):t]),
                        supervision=dict(target,teacher_action=action),
                        audit_only=dict(scene=house,record_id=row['record_id'],physical_route=row['physical_source_route_sha256'],
                            step=t,position=frame['position'],goal=goals[0],geodesic_distance=distance if math.isfinite(distance) else None,
                            source_policy=str(policy_path.relative_to(ROOT)),source_supervision=str(supervision_path.relative_to(ROOT))))
                    policy_payload(sample)
                    stream.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
                    counts['decisions']+=1
                    if target['can_stop'] is None:counts['unknown']+=1
                    elif target['can_stop']:
                        counts['positive']+=1;route_positive+=1
                        counts['positive_with_motion_teacher']+=action!='STOP'
                    else:counts['negative']+=1
                    last_distance=distance
                assert last_distance<3.,'CERTIFIED_TEACHER_TERMINAL_OUTSIDE_GOAL'
                summaries.append(dict(record_id=row['record_id'],house=house,partition=partition,
                    physical_route=row['physical_source_route_sha256'],decisions=len(actions),positive=route_positive,
                    final_distance=last_distance,navmesh_sha256=sha(navmesh),source=row))
    result=dict(status='REAL_FIT_STOP_LABEL_INTERFACE_COMPLETE',routes=6,houses=houses,counts=counts,
        source_index_sha256=sha(SNAPSHOT/'TRAINING_INDEX.jsonl'),official_dataset_sha256=sha(DATASET),
        decisions_sha256=sha(out/'DECISIONS.jsonl'),selection='First three lexicographic FIT houses; first two unique physical routes by record ID per house; chosen without model outcomes',
        partition='First two houses head_fit; third head_check. All houses were in backbone FIT, not blind scenes.',
        old_action_labels_modified=False,model_updates=0,new_environment_decisions=0,renderer_created=False,gpu_hours=0,
        policy_fields=['instruction','rgb_paths','executed_actions'],
        label_scope='can_stop is the existing geodesic<3m criterion at the decision input state, not proof of arbitrary language/program satisfaction',
        training_admission='DATA_INTERFACE_ONLY; no STOP model trained or deployed',records=summaries)
    with (out/'RESULT.json').open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False)
    print({k:v for k,v in result.items() if k!='records'})


if __name__=='__main__':main()
