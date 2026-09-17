"""CPU single-batch strong audit; all failures remain in the observation."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('auto_audit_transport',HERE/'transport.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
GATE=t.WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/gate.py'
GATE_SHA='9aa097b918611537a9d0f2517528a6e257ec6504e860cc53fe28af8c9119fb83'
def audit_main(batch):
    batch=Path(batch);cfg=t.check_inputs(batch)
    run=batch/'run_v1';terminal=json.loads((run/'SUPERVISOR_RESULT.json').read_text())
    assert terminal['cleanup_complete'] is True,'WAIT_FOR_NATURAL_CLEANUP'
    json.loads((run/'LAUNCH_RESULT.json').read_text())
    assert t.sha(GATE)==GATE_SHA
    g=t.load('auto_single_batch_frozen_gate3',GATE)
    out=HERE/'audits'/batch.name;out.mkdir(parents=True,exist_ok=False)
    observation=g.gate.observe_run(run,out)
    accepted=[r for r in observation['attempts'] if r.get('quality_pass') is True and r.get('source_and_phase_binding_verified') is True]
    result={'status':'STRONG_SINGLE_BATCH_AUDIT_COMPLETE','run_root':str(run),'observation':observation,
        'registered_candidates':len(cfg['candidates']),'strongly_accepted_families':len(accepted),
        'same_house_and_hub_variants_correlated':True,'added_to_old_cohort':False,
        'algorithm_gain_claim':False,'scientific_pass':False,'gpu_operations':0}
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='observation'},indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);audit_main(p.parse_args().batch)
