import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'short_revisit_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lock=json.loads((BASE/'run_v1/INPUT_LOCK.json').read_text())
    for p,h in lock.items():assert sha(p)==h,p
    assert json.loads((BASE/'run_v1/SUPERVISOR_RESULT.json').read_text())['cleanup_complete']
    cfg=json.loads((BASE/'run_v1/EXECUTION_CONFIG.json').read_text())
    source=HERE.parent/'short_continuation_cpu/CANDIDATES.json'
    selected=json.loads(source.read_text())['candidates'][1]
    assert len(selected['actions'])==36 and selected['protected_source_frames']==[112,113]
    original=cfg['candidates'][0]['components']['a'];reduced=[]
    for a in original:
        if reduced and a in ('L','R') and reduced[-1] in ('L','R') and a!=reduced[-1]:reduced.pop()
        else:reduced.append(a)
    assert reduced==selected['actions'],'NO_WINDING_FLIP_FREE_REDUCTION'
    cfg['node']='Q35N_SHORT_CONTINUATION_NO_WINDING_FLIP_V3'
    row=cfg['candidates'][0];row['candidate_id']='WF_SHORT_REVISIT_V3_004'
    row['components']['short_a']=selected['actions']
    row['component_provenance'].update(short_proposal_index=1,free_inverse_turn_reduction=True)
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    for p in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',out/'EXECUTION_CONFIG.json',BASE/'run_v1/result.json',BASE/'run_v1/journal/events.jsonl']:
        lock[str(p)]=sha(p)
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump(lock,f,indent=2)
    print(json.dumps(dict(prepared=True,short_loop_actions=36,gpu=2,locked_inputs=len(lock))))
if __name__=='__main__':main()
