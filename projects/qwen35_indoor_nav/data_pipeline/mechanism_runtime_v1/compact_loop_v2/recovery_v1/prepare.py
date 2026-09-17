import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    failed=json.loads((BASE/'run_v1/result.json').read_text())
    assert failed['counts']['actual_actions']==0
    assert 'tracked store must be' in failed['error']['error']
    old=json.loads((BASE/'run_v1/INPUT_LOCK.json').read_text())
    for path,value in old.items():assert sha(path)==value,path
    cfg=json.loads((BASE/'run_v1/EXECUTION_CONFIG.json').read_text())
    cfg['output_scope_recovery']='tracked_store_root_exact_new_recovery_only'
    cfg['supervision_wall_seconds']=2990
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    for path in [*HERE.glob('*.py'),HERE/'AMENDMENT_ZH.md',out/'EXECUTION_CONFIG.json',BASE/'run_v1/result.json']:
        old[str(path)]=sha(path)
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump(old,f,indent=2)
    print(json.dumps(dict(prepared=True,old_attempt_actions=0,remaining_supervision_seconds=2990)))
if __name__=='__main__':main()
