"""Re-run the existing strict physical/semantic audit against historical seals."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data


def main():
    prepared=json.loads((HERE/'DATA.json').read_text())
    quality=data.load('v7_strict_family_audit',data.ASSETS/'quality_cpu/quality.py')
    reports=[]
    for row in prepared['audit']['families']:
        export=data.LINE/row['export']
        family=export.parent.name
        if 'batch_execution_v1' in export.parts:
            batch=next(part for part in export.parts if part.startswith('batch_') and part!='batch_execution_v1')
            original=data.ASSETS/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/special_scale_transport_v1'/batch/family
        else:
            original=data.ASSETS/'diversity_v1r2/audit_v1'/family
        evidence_path=original/'EVIDENCE.json'
        evidence=json.loads(evidence_path.read_text())
        result=quality.audit_family(export,evidence)
        result.update(family_id=family,split=row['split'],source_evidence=str(evidence_path.relative_to(data.LINE)),
            source_evidence_sha256=data.sha(evidence_path),original_training_admission=False,
            current_scope='Authorized exploratory V7 only; no scientific/full-data admission inferred')
        reports.append(result)
        with (HERE/'ASSET_AUDIT_PROGRESS.json').open('w') as stream:
            json.dump(dict(completed=len(reports),total=len(prepared['families']),last_family=family,last_grade=result['grade']),stream,indent=2)
        print(family,result['grade'],result['errors'],flush=True)
    output=dict(status='EXISTING_PHYSICAL_EVIDENCE_REAUDITED',families=reports,
        all_quality_verified=all(row['quality_pass'] and row.get('evidence_verified') for row in reports),
        new_simulation_executions=0,original_training_admission=False,scientific_pass=False)
    with (HERE/'ASSET_REAUDIT.json').open('x') as stream:json.dump(output,stream,indent=2)
    assert output['all_quality_verified'],'REAL_ASSET_AUDIT_FAILED'


if __name__=='__main__':main()
