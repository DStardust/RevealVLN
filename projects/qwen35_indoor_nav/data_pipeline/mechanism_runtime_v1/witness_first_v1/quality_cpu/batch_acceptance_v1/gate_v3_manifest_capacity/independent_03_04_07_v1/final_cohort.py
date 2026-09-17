"""Aggregate all four separately strong-audited members, never favorable subset."""
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
WF=HERE.parents[3]
NAMES=('batch_03r1','batch_04r1','batch_05r1','batch_06r1')

def main():
    path=HERE.parent/'gate.py'
    seal=dict(line.split(None,1)[::-1] for line in (HERE.parent/'SHA256SUMS').read_text().splitlines())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['gate.py']
    spec=importlib.util.spec_from_file_location('final_cohort_original_gate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    reports=[];inputs={}
    for name in NAMES:
        folder=HERE/name
        # No incomplete audit, absent receipt or unsealed report gets promoted.
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            expected,relative=line.split(None,1);p=folder/relative
            assert p.resolve().is_relative_to(folder) and m.gate.old.sha(p)==expected
            inputs[str(p)]=expected
        report=m.gate.old.read(folder/'BATCH_REPORT.json')
        assert Path(report['run_root'])==WF/'batch_execution_v1'/name/'run_v1'
        assert report['exposure_status'] in ('COMPLETE_RUN_INDEPENDENTLY_AUDITED','CLOSED_REJECTED_SOURCE_OR_TERMINAL')
        reports.append(report)
    roots=sorted(p.parent for p in (WF/'batch_execution_v1').glob('batch_*/run_v1/EXECUTION_CONFIG.json'))
    manifest=WF/'multi_program_bank_v1/language_ready_v2/COHORT_V2.json'
    cohort,members,h=m.validate_cohort(manifest,roots)
    assert set(members)=={Path(r['run_root']) for r in reports} and len(members)==4
    result=m.evaluate(reports)
    result.update(evaluation_cohort=cohort,cohort_manifest_sha256=h,all_four_members_retained=True,
        family_audit_input_hashes=inputs,independent_batch_reports=reports,
        batch07_excluded=True,independent_strong_family_audits_reused_without_relabelling=True,
        failed_members_not_replaced=True,source_and_metric_code_unchanged=True,
        all_history_exposure=[{'run_root':str(root),'config_sha256':m.gate.old.sha(root/'EXECUTION_CONFIG.json'),
           'supervisor_result':m.gate.old.read(root/'SUPERVISOR_RESULT.json') if (root/'SUPERVISOR_RESULT.json').exists() else None}
           for root in roots])
    out=HERE/'final_four_member_cohort_v1';out.mkdir(exist_ok=False)
    m.gate.old.save(out/'REPORT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('family_audit_input_hashes','independent_batch_reports','all_history_exposure','evaluation_cohort')},indent=2))

if __name__=='__main__':main()
