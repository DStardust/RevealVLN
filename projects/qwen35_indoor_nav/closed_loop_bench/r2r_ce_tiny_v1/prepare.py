"""CPU-only dataset selection and actual artifact fingerprints; no model evaluation."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import importlib.util

HERE = Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
assert not (HERE/'PROTOCOL.json').exists()
source = c.ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr'
dataset = source/'val_unseen/val_unseen.json.gz'
episodes = json.load(gzip.open(dataset))['episodes']
assert len(episodes) == 1839
house = lambda e: e['scene_id'].split('/')[-2]
salt = 'Q35N_TINY_R2R_20260911:'
order = lambda x: hashlib.sha256((salt+str(x)).encode()).hexdigest()
houses = sorted({house(e) for e in episodes}, key=order)
assert len(houses) == 11
split_path = c.LINE/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json'
split = json.loads(split_path.read_text())
assert not set(houses) & set(split['FIT']+split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
selected = []
for h in houses[:4]:
    es = [e for e in episodes if house(e) == h]
    trajectories = sorted({str(e['trajectory_id']) for e in es}, key=order)[:2]
    for trajectory in trajectories:
        selected.append(sorted([e for e in es if str(e['trajectory_id']) == trajectory],
                               key=lambda e: order(e['episode_id']))[0])
assert len(selected) == 8 and len({(house(e), str(e['trajectory_id'])) for e in selected}) == 8
public = [{k:e[k] for k in ('episode_id','trajectory_id','scene_id','info')} for e in selected]
c.write(HERE/'EPISODES_PRIVILEGED.json', selected, True)
checkpoint = c.TRAIN/'formal/attempt_001/checkpoint_000034800.pt'
receipt = json.loads(Path(str(checkpoint)+'.json').read_text())
assert c.sha(checkpoint) == receipt['sha256'] and receipt['cursor']['updates'] == 34800
train_protocol = c.TRAIN/'PROTOCOL_FILESTORE.json'
tp = json.loads(train_protocol.read_text())
for name, digest in tp['code_sha256'].items():
    assert c.sha(c.TRAIN/name) == digest
lab = c.ROOT/'third_party/habitat-lab'
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=lab,text=True).strip()
assert commit == 'd6ed1c0a0e786f16f261de2beafe347f4186d0d8'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=lab,text=True).strip()
shutil.copyfile(lab/'LICENSE',HERE/'HABITAT_LAB_LICENSE.txt')
assets = [dataset, source/'val_unseen/val_unseen_gt.json.gz',split_path,
          train_protocol,checkpoint,Path(str(checkpoint)+'.json'),
          lab/'habitat/tasks/nav/nav.py',lab/'habitat/core/env.py',lab/'habitat/config/default.py',lab/'LICENSE']
for h in houses[:4]:
    folder = c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{h}'
    assert folder.resolve().is_relative_to(c.ROOT)
    assets.extend(folder/f'{h}{suffix}' for suffix in ('.glb','.navmesh','.house','_semantic.ply'))
for p in assets:
    assert p.resolve().is_relative_to(c.ROOT) and p.is_file()
c.write(HERE/'ASSET_AUDIT.json',dict(dataset=str(dataset),dataset_sha256=c.sha(dataset),
        episodes=1839,houses=houses,selected=public,selection_salt=salt,
        q35n_training_house_overlap=[],legacy_other_research_exposure='UNKNOWN',
        data_variant='local XLM-R preprocessed v1-3 copy; original text and geometry only',
        raw_train_text_geometry_exact_matches=10819,raw_train_count=10819,
        official_download_attempt='TLS EOF: no downloaded assets used',
        habitat_lab_commit=commit,habitat_lab_worktree_clean=True),True)
protocol = dict(id='Q35N_R2R_CE_TINY_V1',checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),
    training_protocol_sha256=c.sha(train_protocol), sample_index_sha256=tp['sample_index_sha256'],
    checkpoint_updates=34800,seed=1109,environment_seed=0,episode_count=8,house_count=4,
    max_steps=500,wall_seconds=2400,gpu_memory_gib=28,cpu_memory_gib=40,output_gib=2,
    gpu=1,gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',rgb_size=224,hfov=90,
    camera_height=1.25,agent_height=1.5,agent_radius=.1,forward_m=.25,turn_deg=15.,
    allow_sliding=True,success_distance=3.,greedy=True,optimizer_updates=0,
    metric_backend='Unmodified official Habitat v0.1.7 class bodies with isolated adapter',
    ndtw=None,sdtw=None,ndtw_reason='Not integrated in this tiny pilot; no proxy substitution',
    main_criterion='Descriptive counts and per-episode outcomes; no scientific performance threshold',
    official_full_trainer_or_leaderboard_run=False)
c.write(HERE/'PROTOCOL.json',protocol,True)
assets += [c.TRAIN/name for name in tp['code_sha256']]
c.write(HERE/'INPUT_BINDINGS.json',{str(p):c.sha(p) for p in assets},True)
print(json.dumps(dict(checkpoint=34800,selected=public),ensure_ascii=False,indent=2))
