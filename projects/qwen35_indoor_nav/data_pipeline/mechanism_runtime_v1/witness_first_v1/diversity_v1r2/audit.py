"""Reuse strict source/phase/27-replay audit; add language and moving-tail gates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import math
import types

from common import HERE, OUT, WF, load, read, save, verify_inputs
from language import ast, parse
build_supervisor = load('diversity_local_supervisor_entry', HERE/'run.py').build_supervisor


def build_auditor():
    original=load('diversity_readonly_resource_audit',WF/'special_scale_transport_v1/audit.py')
    base=original.c.load('diversity_original_gate',original.GATE)
    source=original.adapted_acceptance(base.ADAPTED_SOURCE)
    module=types.ModuleType('diversity_private_quality');module.__file__=str(base.ACCEPTANCE)
    def verify_active(run,cfg,proc):
        path=run/'ACTIVE_ACCOUNTING.jsonl'
        assert path.stat().st_size<=64*1024**2
        raw=path.read_bytes();assert raw.endswith(b'\n')
        supervisor=build_supervisor()
        context=original.verify_events([json.loads(line) for line in raw.splitlines()],supervisor,proc['pid'])
        before=read(run/'GPU_BEFORE.json')
        supervisor.check_gpu(dict(before,processes={int(p):v for p,v in before['processes'].items()}))
        assert not before['processes'] and before['utilization']==0
        assert read(run/'LAUNCH_RESULT.json')['idle_final_guard_unchanged'] is True
        return context
    module.verify_active_log=verify_active;module.verify_resource=original.verify_resource
    exec(compile(source,str(HERE/'audit.py')+'::sealed_acceptance_adapter','exec'),module.__dict__)
    module.HERE=HERE/'audit_v1'
    return module


def main():
    cfg=verify_inputs()
    auditor=build_auditor();destination=HERE/'audit_v1';destination.mkdir(exist_ok=False)
    reports=[]
    for row in cfg['candidates']:
        aid=row['candidate_id'];folder=OUT/'bundles'/aid
        result=read(folder/'result.json')
        if not result.get('physical_certified'):
            reports.append(dict(attempt_id=aid,house_id=row['house_id'],quality_pass=False,
                                physical_result=result));continue
        report=auditor.build_family_evidence(OUT,aid,destination/aid)
        if report.get('quality_pass'):
            manifest=read(folder/'export_v4/MANIFEST.json')
            for task_id,task in manifest['compiler_config']['tasks'].items():
                assert parse(task['instruction'],row['roles'])==ast(row['roles'],row['tasks'][task_id])
            candidate=manifest['candidate'];k=len(candidate['histories']['H_A'])
            trace=read(folder/'export_v4/traces/t0_0.json')
            before=trace['observations'][k-8]['pose']['position']
            after=trace['observations'][k]['pose']['position']
            distance=math.dist(before,after)
            assert candidate['public_tail']=='FFFFFFFF' and 1.95<=distance<=2.05
            report.update(language_ast_verified=True,actual_common_tail_displacement_m=distance,
                          actual_decision_hub=after,source_house_is_new=False,
                          ce_unique_owners=manifest['ce_unique_owners'],
                          grade='QUALITY_VERIFIED_FIT_DIVERSITY_V1_NOT_MODEL_GAIN')
        save(destination/aid/'DIVERSITY_REPORT.json',report)
        reports.append(report)
    accepted=[r for r in reports if r.get('quality_pass')]
    result=dict(registered_candidates=len(cfg['candidates']),attempts=reports,
                accepted_families=len(accepted),accepted_houses=len({r['house_id'] for r in accepted}),
                new_houses=0,cells=len(accepted)*18,
                small_batch_acceptance=len({r['house_id'] for r in accepted})>=2,
                training_admission=False,scientific_pass=False)
    save(destination/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='attempts'},indent=2))


if __name__=='__main__': main()

