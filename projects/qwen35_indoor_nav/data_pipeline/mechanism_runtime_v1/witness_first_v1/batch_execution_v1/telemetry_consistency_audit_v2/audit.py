"""V1 criteria unchanged; fix only exclusive writer's intended output scope."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
V1=HERE.parent/'telemetry_consistency_audit_v1/audit.py'
V1_SHA='d35266eeb0c5b88f44dcb96cbb5176ac478512e053eeb82baf880ef57e3048f3'

def _write_report(path,result):
    path=Path(path).absolute()
    if path!=path.resolve() or not path.is_relative_to(HERE) or path.name!='REPORT.json':
        raise ValueError('REPORT_OUTPUT_SCOPE')
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)

def adapted_source(source):
    if hashlib.sha256(source.encode()).hexdigest()!=V1_SHA:raise ValueError('SEALED_V1_CHANGED')
    old="output.mkdir(exist_ok=False);old.save(output/'REPORT.json',result)"
    new="output.mkdir(exist_ok=False);_write_report(output/'REPORT.json',result)"
    if source.count(old)!=1:raise ValueError('EXACT_WRITER_REPLACEMENT')
    result=source.replace(old,new)
    if result.replace(new,old)!=source:raise ValueError('NONREVERSIBLE_TRANSFORM')
    return result

exec(compile(adapted_source(V1.read_text()),str(V1)+'::exclusive_writer_v2','exec'),globals())
