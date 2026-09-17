import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'revisit_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    lock=json.loads((BASE/'run_v1/INPUT_LOCK.json').read_text())
    for p,h in lock.items():assert sha(p)==h,p
    assert json.loads((BASE/'run_v1/SUPERVISOR_RESULT.json').read_text())['cleanup_complete']
    proposals=HERE.parent/'short_continuation_cpu/CANDIDATES.json'
    source=json.loads(proposals.read_text())
    for p,h in source['source_hashes'].items():assert sha(p)==h,p
    selected=source['candidates'][0]
    assert len(selected['actions'])==34 and selected['protected_source_frames']==[16,17]
    cfg=json.loads((BASE/'run_v1/EXECUTION_CONFIG.json').read_text())
    cfg['node']='Q35N_SHORT_CONTINUATION_COMPLETED_SUBGOAL_REVISIT_V2'
    cfg['candidates'][0]['candidate_id']='WF_SHORT_REVISIT_004'
    cfg['candidates'][0]['components']['short_a']=selected['actions']
    cfg['candidates'][0]['component_provenance'].update(short_proposal_sha256=sha(proposals),short_proposal_index=0,
        original_history_components_unchanged=True)
    assert len(cfg['candidates'][0]['components']['a'])==116
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    lock.update(source['source_hashes'])
    for p in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',proposals,out/'EXECUTION_CONFIG.json',BASE/'run_v1/result.json']:
        lock[str(p)]=sha(p)
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump(lock,f,indent=2)
    print(json.dumps(dict(prepared=True,gpu=2,selected_short_actions=34,locked_files=len(lock))))
if __name__=='__main__':main()
