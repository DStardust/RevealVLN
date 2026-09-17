import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'assembly_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lock=json.loads((BASE/'run_v1/INPUT_LOCK.json').read_text())
    for p,h in lock.items():assert sha(p)==h,p
    terminal=json.loads((BASE/'run_v1/SUPERVISOR_RESULT.json').read_text());assert terminal['cleanup_complete']
    cfg=json.loads((BASE/'run_v1/EXECUTION_CONFIG.json').read_text())
    cfg['node']='Q35N_WITNESS_COMPLETED_SUBGOAL_REVISIT_V1'
    cfg['control_type']='completed_subgoal_revisit_placement_not_event_free_detour'
    cfg['candidates'][0]['candidate_id']='WF_REVISIT_004'
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    for p in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',out/'EXECUTION_CONFIG.json',BASE/'run_v1/result.json']:
        lock[str(p)]=sha(p)
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump(lock,f,indent=2)
    print(json.dumps(dict(prepared=True,gpu=2,control_type=cfg['control_type'])))
if __name__=='__main__':main()
