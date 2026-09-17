"""CPU-only parser/schema checks; deliberately NOT family/run acceptance."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE.parent/'quality_cpu/batch_acceptance_v1/acceptance.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='a2cae535b132b652c289d753bcf11dc2f55413bb942fd731528198a8ab4fa7d8'
spec=importlib.util.spec_from_file_location('clock_compat_sealed_acceptance',path)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def inspect(root):
    root=Path(root).resolve()
    assert root.is_relative_to(HERE)
    checks=[]
    for folder in sorted(root.glob('*/journal')):
        raw=(folder/'events.jsonl').read_bytes();head=json.loads((folder/'HEAD.json').read_text())
        config=json.loads(raw.splitlines()[0])['payload']
        rows=old.parse_journal(raw,head,config)
        budgets=[r['payload'] for r in rows if r['kind']=='budget']
        for state in budgets:
            old.quality.bridge.BudgetLedger(state['limits'],state=state,clock=lambda:state['last_clock'])
        checks.append({'journal':str(folder.relative_to(HERE)),'canonical_chain_head_pass':True,
                       'original_budget_schema_states_passed':len(budgets)})
    assert len(checks)==6
    return {'status':'CPU_PARSER_SCHEMA_COMPATIBLE_ONLY','journals':checks,
            'runtime_or_family_acceptance':False,'source_lock_switch_authorized':False,
            'full_runtime_auditor_pass_not_tested':True}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).absolute();assert out==out.resolve() and out.is_relative_to(HERE)
    report=inspect(a.benchmark)
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
