"""One-shot full dataset, latest receipt checkpoint, CPU validation and freeze."""
import collections
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
    assert not (HERE/'PROTOCOL.json').exists() and not (HERE/'SOURCE_LOCK.json').exists()
    sources=sorted(HERE.glob('*.py'))
    for path in sources:compile(path.read_text(),str(path),'exec')
    test=subprocess.run([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(HERE/'test_full.py')],
        cwd=c.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    c.write(HERE/'CPU_TEST_RESULT.json',dict(unix=time.time(),passed=test.returncode==0,returncode=test.returncode,output=test.stdout),True)
    assert test.returncode==0,test.stdout
    old_lock=c.old.verify_lock();files=dict(old_lock['files'])
    source=c.ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen'
    dataset=source/'val_unseen.json.gz';gt_path=source/'val_unseen_gt.json.gz'
    episodes=json.load(gzip.open(dataset))['episodes'];gt=json.load(gzip.open(gt_path))
    assert len(episodes)==1839 and len({str(e['episode_id']) for e in episodes})==1839
    assert all(str(e['episode_id']) in gt and gt[str(e['episode_id'])]['locations'] for e in episodes)
    houses=sorted({e['scene_id'].split('/')[-2] for e in episodes});assert len(houses)==11
    split=json.loads((c.LINE/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json').read_text())
    assert not set(houses)&set(split['FIT']+split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    import hashlib
    order=lambda x:hashlib.sha256(('Q35N_FULL_R2R_20260911:'+str(x)).encode()).hexdigest()
    episodes.sort(key=lambda e:(order(e['scene_id'].split('/')[-2]),order(e['episode_id'])))
    c.write(HERE/'EPISODES_PRIVILEGED.json',episodes,True)
    # Freeze once at preparation time, not once per worker or once per episode.
    observed_unix=time.time();receipts=sorted((c.TRAIN/'formal/attempt_001').glob('checkpoint_*.pt.json'))
    receipt_path=receipts[-1];checkpoint=Path(str(receipt_path)[:-5]);receipt=json.loads(receipt_path.read_text())
    assert c.sha(checkpoint)==receipt['sha256']
    updates=receipt['cursor']['updates'];tp_path=c.TRAIN/'PROTOCOL_FILESTORE.json';tp=json.loads(tp_path.read_text())
    for name,digest in tp['code_sha256'].items():assert c.sha(c.TRAIN/name)==digest
    p=json.loads((c.TINY/'PROTOCOL.json').read_text())
    for key in ('ndtw','sdtw','ndtw_reason'):p.pop(key,None)
    p.update(id='Q35N_R2R_CE_FULL_V1',checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),checkpoint_updates=updates,
        training_protocol_sha256=c.sha(tp_path),sample_index_sha256=tp['sample_index_sha256'],
        episode_count=1839,house_count=11,houses=houses,house_counts=dict(collections.Counter(e['scene_id'].split('/')[-2] for e in episodes)),
        lanes=8,wall_seconds=64800,cpu_memory_gib=64,output_gib=8,gt_path=str(gt_path),
        main_criterion='Full 1839-episode descriptive evaluation; no scientific performance PASS threshold',
        ndtw_backend='exact Euclidean DTW, official FDTW=False semantics',
        previous_tiny_episode_ids=[1085,1333,638,426,1485,651,664,944],
        checkpoint_selection='highest update count with completed receipt observed once before source freeze',
        checkpoint_selected_unix=observed_unix,parity_threshold=dict(max_abs=.15,relative_l2=.03,all_argmax_identical=True),
        parity_fallback_batch_size=1,parity_pass_batch_size=8)
    c.write(HERE/'PROTOCOL.json',p,True)
    c.write(HERE/'CHECKPOINT_SELECTION.json',dict(unix=observed_unix,selected=str(checkpoint),receipt=receipt,
        latest_observed_receipt=str(receipt_path),total_completed_receipts=len(receipts)),True)
    fixtures=[[],[]]
    for i in range(8):
        e=json.loads((c.TINY/'run_001'/f'episode_{i:02d}.json').read_text())
        first=c.TINY/'run_001/frames'/f'ep_{i:02d}_step_000.png'
        last=c.TINY/'run_001/frames'/f'ep_{i:02d}_step_{e["steps"]:03d}.png'
        for path in (first,last,c.TINY/'run_001'/f'episode_{i:02d}.json'):files[str(path)]=c.sha(path)
        fixtures[0].append(dict(instruction=e['instruction'],images=[str(first)],executed=[]))
        history=[c.ACTIONS[j%3] for j in range(i+1)]
        fixtures[1].append(dict(instruction=e['instruction'],images=[str(first),str(last)] if i%2 else [str(last)],executed=history))
    c.write(HERE/'PARITY_FIXTURES.json',fixtures,True)
    for h in houses:
        folder=c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{h}'
        for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
            path=folder/f'{h}{suffix}';assert path.resolve().is_relative_to(c.ROOT);files[str(path)]=c.sha(path)
    for path in (dataset,gt_path,checkpoint,receipt_path,tp_path,c.TINY/'SOURCE_LOCK.json'):
        files[str(path)]=c.sha(path)
    c.write(HERE/'ASSET_AUDIT.json',dict(source=str(dataset),episodes=1839,house_count=11,houses=houses,
        house_overlap_current_training=[],source_variant='local preprocessed v1-3 original texts/geometry',
        source_provenance=c.sha(c.TINY/'ASSET_AUDIT.json'),legacy_other_exposure='UNKNOWN',foundation_exposure='UNKNOWN',
        prior_tiny_exposed_episodes=8,remaining_episodes=1831),True)
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    git_dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=c.ROOT,text=True)
    c.write(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),decision='APPROVED_FULL_FIXED_BASELINE_EVALUATION',
        source_review='sequential main-agent interface/resource/measurement review, not independent human review',
        cpu_passed=True,gpu_batch_gate_required=True,optimizer_updates=0,scientific_novelty_claim=False,
        git_commit=git_commit,tracked_dirty=git_dirty,foreign_process_signals_allowed=False,automatic_retry=False),True)
    auth=c.LINE/'authorizations/ORDINARY_NAVBENCH_FULL_V1_20260911.json';files[str(auth)]=c.sha(auth)
    for path in HERE.iterdir():
        if path.is_file():files[str(path)]=c.sha(path)
    c.write(HERE/'SOURCE_LOCK.json',dict(unix=time.time(),files=files),True)
    print(json.dumps(dict(checkpoint_updates=updates,checkpoint=str(checkpoint),episodes=1839,houses=11,locked_files=len(files),CPU_passed=True)))


if __name__=='__main__':main()
