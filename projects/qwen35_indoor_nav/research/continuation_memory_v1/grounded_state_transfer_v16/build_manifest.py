"""Deterministic asset-only split; no candidate policy scores are read."""
import collections
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)

def main():
    planner = load('v16_semantic_metadata', LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    scenes = ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d'
    inventory = {p.name:p for p in scenes.iterdir() if p.is_dir()}
    exposure = collections.defaultdict(list)
    sources = sorted(HERE.parent.glob('*/DATA.json'))
    sources += [HERE.parent/'semantic_transfer_v12/POOL_PROTOCOL.json']
    for path in sources:
        data = read(path)
        for house in set(strings(data)) & inventory.keys():
            exposure[house].append(str(path.relative_to(LINE)))
    previous=read(V15/'NEW_HOUSE_PROTOCOL.json')
    for house in previous['excluded_exposed_houses']+[r['house_id'] for r in previous['houses']]:
        exposure[house].append('research/continuation_memory_v1/legal_closed_loop_v15/NEW_HOUSE_PROTOCOL.json: exposed or actually scouted')
    ordinary = read(HERE.parent/'natural_transfer_v9/DATA.json')
    ordinary_rows = ordinary['records']
    assert all(r['row']['scene_group'] in ordinary['fit_houses'] for r in ordinary_rows if r['partition']=='fit')
    eligible = []
    rejected = []
    signatures = collections.Counter()
    for house, folder in sorted(inventory.items()):
        required = [folder/(house+suffix) for suffix in ('.glb','.house','.navmesh','_semantic.ply')]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            rejected.append(dict(house=house, reason='MISSING_SCENE_ASSET', paths=missing)); continue
        groups = planner.semantic_groups(planner.parse_house((folder/(house+'.house')).read_text()))
        if house in exposure or len(groups)<3:
            rejected.append(dict(house=house, reason='EXPOSED_HOUSE' if house in exposure else 'INSUFFICIENT_ROLES', sources=exposure[house])); continue
        roles = [dict(signature=list(k), spec=planner.role_spec(k), eligible=[r['object_index'] for r in v]) for k,v in sorted(groups.items())]
        signatures.update(set(tuple(r['signature'][:2]) for r in roles))
        eligible.append(dict(house=house, scene=str(required[0]), roles=roles, assets=[str(p) for p in required]))
    # Prefer semantic support shared across available houses, then a fixed hash.
    eligible.sort(key=lambda r:(-sum(signatures[tuple(x['signature'][:2])]>=6 for x in r['roles']), digest(['V16_SPLIT',r['house']])))
    if len(eligible)<9:
        write(HERE/'MANIFEST_BLOCKED.json',dict(reason='INSUFFICIENT_UNEXPOSED_REAL_HOUSES',available=len(eligible),required=9,rejected=rejected),True)
        raise RuntimeError('INSUFFICIENT_UNEXPOSED_REAL_HOUSES')
    chosen = eligible[:9]
    for i,row in enumerate(chosen):
        row['split'] = 'FIT' if i<4 else 'DEV' if i==4 else 'TEST'
        row['target_families'] = 4 if i<4 else 2
        row['asset_sha256'] = {p:sha(Path(p)) for p in row.pop('assets')}
        row['base_training_exposure'] = 'not_resolved_from_ordinary_index; no base-unseen claim'
    manifest = dict(version='V16', houses=chosen, rejected=rejected,
        available_unexposed_houses=[r['house'] for r in eligible], exposure=dict(exposure),
        selection='semantic support count then SHA256(V16_SPLIT,house), no policy or navigation score',
        test_freeze='Asset candidates preregistered now; physical family/condition manifest sealed after FIT/DEV pilot and before any model TEST score.',
        ordinary_data_sha256=sha(HERE.parent/'natural_transfer_v9/DATA.json'),
        template_families=dict(FIT='first_then',DEV='prerequisite_before_stop',TEST='completion_order'),
        original_training_admission_modified=False)
    immutable(HERE/'DATA_MANIFEST.json',manifest)
    old=read(V15/'FEATURE_PROTOCOL.json')
    protocol=dict(version='V16.0',baseline_commit=BASELINE,gpu=1,gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',
        checkpoint=str(LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt'),
        checkpoint_sha256='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775',
        model_source_sha256='89c2aac37d0d0519c01f51acf8f19b42087434f42e4cf56fc76fd01cf5297389',
        seed=1209,seeds=[1209,1210,1211],arms=['B1','B2','Ours'],steps_per_arm=1200,checkpoint_every=200,
        learning_rate=.001,weight_decay=.01,memory_slots=8,memory_width=64,feature_width=2048,
        model_memory_gib=25,process_rss_gib=64,artifact_gib=40,min_free_gpu_gib=2,shared_gpu=True,
        gpu_session_hours=16,max_session_hours=3,infra_retries=3,
        warmup_indices=8,warmup_repeats=2,golden_inputs=16,
        collection=dict(hubs_per_house=12,roles_per_hub=8,max_proposals_per_house=64,
                        history_cap=240,suffix_cap=240,full_budget=500,environment_seed=0,
                        perturbation=['R']*6+['F'],perturb_at='after anchor witness before return to registered hub',
                        teacher='fixed greedy geodesic follower, no learned model',
                        suffix_programs=['terminal','anchor_A then terminal','anchor_B then terminal']),
        auxiliary_weights='FIT inverse class frequency normalized by actual weighted valid examples; fixed before training',
        loss_weights=dict(action=1.,fork=1.,ordinary=1.,preservation_kl=1.,auxiliary=1.),
        preservation_steps=156,selector='argmax(method_logits), index tie break; no native STOP guard',
        primary='safe_v16 PASS / all planned main conditions, equal house and seed; Ours minus B2',
        planned=dict(families=26,physical_executions=312,cross_labels=936,models=9,conditions=80,rollouts=720,main_per_arm=192,prefix_comparisons=480),
        delta_threshold=.10,test_selection_from_scores=False,base_updates=0,
        resource_note='16h planning upper bound including failures, revised only using FIT/DEV actual throughput before TEST freeze. Shared-device timing is not exclusive deployment performance.',
        data_manifest_sha256=sha(HERE/'DATA_MANIFEST.json'))
    config=read(V15/'features_run_001/CONFIG.json')
    for key in ('training_protocol_sha256','sample_index_sha256'):
        protocol[key]=config[key]
    immutable(HERE/'PROTOCOL.json',protocol)
    paths=[LINE/'.envs/q35n_qwen_g2_v1/bin/python3', LINE/'.envs/q35n_habitat_v017_g0r/bin/python3',
           ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3', Path(protocol['checkpoint']),
           LINE/'runtime/models/Qwen3.5-2B_15852e8/config.json']
    checks=[dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None) for p in paths]
    catalog=read(LINE/'reviews/Q35N_V15_HANDOFF_20260920/ARTIFACT_CATALOG.json')
    olddata=read(V15/'DATA.json')
    refs=list(olddata['contents'].values())
    checks.extend(dict(path=str(LINE/r['line_relative_path']),exists=(LINE/r['line_relative_path']).is_file()) for r in refs[:2])
    audit=dict(dependencies=checks,checkpoint_hash=sha(Path(protocol['checkpoint'])),
               model_source_hash=sha(LINE/'sft_acceptance/ordinary_sync_recovery_v1/model.py'),
               baseline_source_diff=subprocess.check_output(['git','diff',BASELINE,'--stat','--',str(LINE.relative_to(ROOT))],cwd=ROOT,text=True),
               ordinary_fit_houses=ordinary['fit_houses'],ordinary_check_houses=ordinary['check_houses'],
               ordinary_routes=len(ordinary_rows),new_houses=[r['house'] for r in chosen],
               house_overlap_with_ordinary=set(ordinary['fit_houses']) & {r['house'] for r in chosen},
               old_artifact_catalog_entries=len(catalog['uploaded_artifacts']))
    audit['house_overlap_with_ordinary']=sorted(audit['house_overlap_with_ordinary'])
    immutable(HERE/'LOCAL_ASSET_AUDIT.json',audit)
    if not all(r['exists'] for r in checks):raise RuntimeError('LOCAL_DEPENDENCY_MISSING')
    print(json.dumps(dict(houses=[(r['split'],r['house']) for r in chosen],available=len(eligible)),indent=2))

if __name__=='__main__':main()
