"""Read-only targeted production observations; sealed capacity-v3 auditor."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
WF=HERE.parents[3]
BE=WF/'batch_execution_v1'
NAMES=('batch_03r1','batch_04r1','batch_07')

def save(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def observe():
    report={'observed_unix':time.time(),'batches':{},'gpu_operations':0,
            'cohort_stability_claim':False,'scientific_pass':False}
    for name in NAMES:
        run=BE/name/'run_v1';row={}
        for filename in ('PROGRESS.json','result.json','SUPERVISOR_RESULT.json','LAUNCH_RESULT.json'):
            p=run/filename
            if p.is_file():
                try:
                    raw=p.read_bytes();row[filename]=json.loads(raw)
                    row[filename+'_sha256']=hashlib.sha256(raw).hexdigest()
                except (OSError,ValueError) as exc:row[filename+'_observation_error']=repr(exc)
        row['closed_evidence_present']=all(k in row for k in ('result.json','SUPERVISOR_RESULT.json','LAUNCH_RESULT.json'))
        final=HERE/name/'BATCH_REPORT.json'
        if final.is_file():row['independent_audit']=json.loads(final.read_text())
        report['batches'][name]=row
    return report

def audit_closed(report):
    source=HERE.parent/'gate.py'
    checks=dict(line.split(None,1)[::-1] for line in (HERE.parent/'SHA256SUMS').read_text().splitlines())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==checks['gate.py']
    s=importlib.util.spec_from_file_location('independent_capacity3',source)
    adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
    for name,row in report['batches'].items():
        if not row['closed_evidence_present'] or 'independent_audit' in row:continue
        target=HERE/name
        target.mkdir(exist_ok=False)
        result=adapter.gate.observe_run(BE/name/'run_v1',target)
        save(target/'BATCH_REPORT.json',result)
        # Individual batch eligibility is not the predeclared four-batch cohort.
        verdict=adapter.evaluate([result])
        save(target/'INDIVIDUAL_BATCH_ONLY.json',dict(per_batch=verdict['per_batch'],
             total_family_quality_pass=sum(f.get('quality_pass') is True for f in result['attempts']),
             cohort_stability_claim=False,scientific_pass=False))
        row['independent_audit']=result
    return report

if __name__=='__main__':
    report=audit_closed(observe())
    save(HERE/('OBSERVATION_'+str(time.time_ns())+'.json'),report)
    summary={name:{'progress':r.get('PROGRESS.json'),'closed':r['closed_evidence_present'],
        'audited_status':r.get('independent_audit',{}).get('exposure_status'),
        'source_errors':r.get('independent_audit',{}).get('source_errors'),
        'quality_pass':sum(f.get('quality_pass') is True for f in r.get('independent_audit',{}).get('attempts',[]))}
        for name,r in report['batches'].items()}
    print(json.dumps(summary,indent=2))
