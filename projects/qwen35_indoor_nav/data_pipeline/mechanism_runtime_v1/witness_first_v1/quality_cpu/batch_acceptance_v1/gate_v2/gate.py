"""At least two batches; all other sealed V1 thresholds remain unchanged.

No favorable subset is selected automatically. Every supplied batch remains in
the verdict and exposure ledger. Recovered content is separate inventory only.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE.parent/'acceptance.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!='a2cae535b132b652c289d753bcf11dc2f55413bb942fd731528198a8ab4fa7d8':
    raise ValueError('ORIGINAL_AUDITOR_CHANGED')
spec=importlib.util.spec_from_file_location('gate_v2_original',path)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
require=old.require

def evaluate(batches,recovered=()):
    errors=[];qualified=[];counts=Counter();per=[]
    if len(batches)<2:errors.append('NEED_AT_LEAST_TWO_PREDECLARED_BATCHES')
    for b in batches:
        current=[];attempts=b.get('attempts',[])
        if b.get('bindings_verified') is not True:current.append('UNVERIFIED_COMPLETE_RUN_BINDINGS')
        if b.get('all_frozen_candidates_attempted_and_terminal') is not True:current.append('FROZEN_ATTEMPT_DENOMINATOR_INCOMPLETE')
        passed=[r for r in attempts if r.get('quality_pass') is True and r.get('source_and_phase_binding_verified') is True]
        nh=len(old.independent_hubs(attempts));nq=len(old.independent_hubs(passed))
        if nh<3:current.append('BATCH_NEEDS_THREE_ATTEMPTED_DISTINCT_HUBS')
        if nq<2:current.append('BATCH_NEEDS_TWO_QUALIFIED_DISTINCT_HUBS')
        per.append({'run_root':b['run_root'],'attempted_independent_hubs':nh,'qualified_independent_hubs':nq,
                    'single_batch_eligible':not current,'errors':current})
        errors.extend(b['run_root']+':'+e for e in current)
        # Ineligible/unverified runs do not donate nominal PASS rows to the total.
        if not current:qualified.extend(passed);counts.update(r['control_type'] for r in passed)
    hubs=old.independent_hubs(qualified);houses={r['house_id'] for r in qualified}
    if len(hubs)<6:errors.append('NEED_SIX_QUALIFIED_PHYSICAL_HUBS')
    if len(houses)<3:errors.append('NEED_THREE_FIT_HOUSES')
    combined=qualified+list(recovered)
    return {'version':'AT_LEAST_TWO_BATCHES_V2','preliminary_cross_house_production_pass':not errors,
        'scope':'initial_cross_house_repeated_production_only','all_supplied_batches_in_verdict':True,
        'batch_count':len(batches),'per_batch':per,'distinct_complete_run_qualified_hubs':len(hubs),'complete_run_fit_houses':len(houses),
        'control_type_counts':dict(counts),'errors':errors,
        'recovery_inventory':{'families':len(recovered),'distinct_hubs':len(old.independent_hubs(recovered)),
            'fit_houses':len({r['house_id'] for r in recovered}),'used_for_batch_gate':False},
        'combined_data_inventory_only':{'distinct_hubs':len(old.independent_hubs(combined)),
            'fit_houses':len({r['house_id'] for r in combined}),'not_a_batch_gate':True},
        'statistical_stability_pass':False,'model_generalization_pass':False,'scientific_pass':False}

def recovery_inventory(index_path):
    rows=[];path=old.path_safe(index_path)
    for line in path.read_text().splitlines():
        row=json.loads(line);report_path=old.path_safe(row['quality_report'])
        require(old.sha(report_path)==row['quality_report_sha256'],'RECOVERY_REPORT_CHANGED')
        report=old.read(report_path)
        require(report['recovery_content_pass'] is True and report['quality_pass'] is False,'NOT_SEPARATELY_GRADED_RECOVERY')
        require(row['fit_data_admitted_by_main'] is True and row['actual_training_authorized'] is False,'RECOVERY_ADMISSION_SCOPE')
        require(report['semantics']['house_id']==row['house_id'] and report['semantics']['hub_position']==row['hub_position'],'RECOVERY_INDEX_IDENTITY')
        require(old.sha(Path(row['family_root'])/'MANIFEST.json')==row['manifest_sha256'],'RECOVERY_MANIFEST_CHANGED')
        rows.append({'candidate_id':row['family_id'],'house_id':row['house_id'],'hub_position':row['hub_position'],
            'source_grade':report['grade'],'quality_pass':False,'recovery_content_pass':True,'training_authorized':False})
    require(len({r['candidate_id'] for r in rows})==len(rows),'DUPLICATE_RECOVERY_INDEX_ID')
    return rows

def observe_run(run_root,out):
    run=old.path_safe(run_root);cfg=old.read(run/'EXECUTION_CONFIG.json')
    row={'run_root':str(run),'bindings_verified':False,'all_frozen_candidates_attempted_and_terminal':False,
         'attempts':[],'frozen_candidate_ids':[r['candidate_id'] for r in cfg['candidates']],
         'source_errors':[],'exposure_status':'ACTIVE_OR_PREWORKER_NO_TERMINAL'}
    ids=set(row['frozen_candidate_ids']);actual=set()
    if (run/'journal/HEAD.json').is_file():
        try:
            head=old.read(run/'journal/HEAD.json');raw=(run/'journal/events.jsonl').read_bytes()
            require(0<head['byte_length']<=len(raw),'HEAD_BYTES')
            records=old.parse_journal(raw[:head['byte_length']],head,cfg)
            for rec in records:
                if rec['kind']=='budget':actual.update(rec['payload']['bundles'])
            row['uncommitted_tail_bytes']=len(raw)-head['byte_length']
            row['head_committed_prefix_verified_for_exposure']=True
        except (KeyError,ValueError,OSError,TypeError) as e:row['source_errors'].append('EXPOSURE_JOURNAL:'+repr(e))
    for c in cfg['candidates']:
        if c['candidate_id'] in actual:
            row['attempts'].append({'candidate_id':c['candidate_id'],'house_id':c['house_id'],
                'hub_position':c['configuration']['u_position'],'quality_pass':False,'control_type':cfg.get('control_type')})
    for name in ('LEASE_RESULT.json','LAUNCH_RESULT.json','SUPERVISOR_RESULT.json'):
        if (run/name).is_file():row[name]=old.read(run/name)
    if not (run/'SUPERVISOR_RESULT.json').is_file():return row
    row['exposure_status']='CLOSED_PENDING_SOURCE_ACCEPTANCE'
    try:
        _,cfg,records,source=old.inspect_run(run)
        require(set(source['budget']['bundles'])==ids,'FROZEN_ATTEMPT_SET_NOT_COMPLETE')
        require({r['candidate_id'] for r in source['source_worker_result']['bundles']}==ids,'RUN_RESULTS_OMIT_ATTEMPTS')
        row['all_frozen_candidates_attempted_and_terminal']=True;row['bindings_verified']=True
        row['attempts']=[]
        for c in cfg['candidates']:
            report=old.build_family_evidence(run,c['candidate_id'],out/c['candidate_id'])
            report.setdefault('house_id',c['house_id']);report.setdefault('hub_position',c['configuration']['u_position'])
            report.setdefault('control_type',cfg.get('control_type'));row['attempts'].append(report)
        row['exposure_status']='COMPLETE_RUN_INDEPENDENTLY_AUDITED'
    except (KeyError,ValueError,OSError,TypeError,AssertionError) as e:
        row['exposure_status']='CLOSED_REJECTED_SOURCE_OR_TERMINAL';row['source_errors'].append(repr(e))
    return row

def validate_cohort(manifest_path,all_run_roots):
    path=old.path_safe(manifest_path);manifest=old.read(path);h=old.sha(path)
    require(manifest['schema_version']=='q35n.production_cohort.v2','COHORT_VERSION')
    require(manifest['criteria_version']=='AT_LEAST_TWO_BATCHES_V2','COHORT_CRITERIA')
    require(bool(manifest['cohort_id']) and bool(manifest['mechanism_version']) and bool(manifest['transport_version']),'COHORT_PROTOCOL_MISSING')
    roots=[old.path_safe(p) for p in manifest['declared_run_roots']]
    require(len(roots)>=2 and len(roots)==len(set(roots)),'COHORT_MEMBER_DENOMINATOR')
    require(set(roots)<=set(all_run_roots),'COHORT_MEMBER_NOT_IN_EXPOSURE')
    for run in roots:
        lock=old.read(run/'INPUT_LOCK.json');proc=old.read(run/'PROCESS.json')
        require(lock.get(str(path))==h,'COHORT_NOT_IN_PREACTION_INPUT_LOCK')
        require(path.stat().st_mtime<=proc['started_unix'] and (run/'INPUT_LOCK.json').stat().st_mtime<=proc['started_unix'],'COHORT_NOT_PREACTION')
    return manifest,roots,h

def audit_all(batch_root,output,recovery_index=None,cohort_manifest=None):
    root=old.path_safe(batch_root);out=old.path_safe(output)
    require(out.is_relative_to(HERE) and out!=HERE,'OUTPUT_SCOPE');out.mkdir(parents=True,exist_ok=False)
    # Discover all declared batch_* roots, not just successful ones. Scope and
    # file hashes are captured at this audit time; active rows cannot receive PASS.
    roots=sorted(p.parent for p in root.glob('batch_*/run_v1/EXECUTION_CONFIG.json'))
    require(len(roots)==len(set(roots)) and len(roots)<=128,'BATCH_ENUMERATION')
    exposure=[];seals={}
    for index,run in enumerate(roots):
        for name in ('EXECUTION_CONFIG.json','INPUT_LOCK.json'):
            if (run/name).is_file():seals[str(run/name)]=old.sha(run/name)
        target=out/('batch_'+str(index));target.mkdir()
        exposure.append(observe_run(run,target))
    recovered=recovery_inventory(recovery_index) if recovery_index else []
    if recovery_index:seals[str(old.path_safe(recovery_index))]=old.sha(recovery_index)
    # No retrospective cohort is invented from whichever historic runs passed.
    # Future cohorts may supersede failed engineering versions, provided every
    # declared member was fixed together before its runtime and remains included.
    cohort=None;cohort_error=None;selected=exposure
    if cohort_manifest:
        try:
            cohort,selected_roots,h=validate_cohort(cohort_manifest,roots)
            selected=[b for b in exposure if Path(b['run_root']) in selected_roots]
            seals[str(old.path_safe(cohort_manifest))]=h
        except (KeyError,ValueError,OSError,TypeError) as e:cohort_error=repr(e)
    result=evaluate(selected,recovered)
    if cohort is None:
        result['preliminary_cross_house_production_pass']=False
        result['errors'].append('NO_VALID_PREACTION_EVALUATION_COHORT')
    complete_content=[f for b in exposure if b['bindings_verified'] for f in b['attempts']
                      if f.get('quality_pass') is True and f.get('source_and_phase_binding_verified') is True]
    result['all_history_data_inventory_only']={'complete_run_families':len(complete_content),
        'recovered_families':len(recovered),'combined_distinct_hubs':len(old.independent_hubs(complete_content+recovered)),
        'combined_FIT_houses':len({f['house_id'] for f in complete_content+recovered}),
        'does_not_supply_batch_gate':True}
    result.update(batch_exposure=exposure,declared_run_roots=list(map(str,roots)),input_hashes=seals,
        evaluation_cohort=cohort,evaluation_cohort_error=cohort_error,
        historical_failures_retained_but_not_permanent_veto_of_future_predeclared_cohort=True,
        old_v1_exact_two_implementation_unchanged=True,
        criterion_change='len(batches)==2 -> len(batches)>=2 only; every included batch still must meet all original per-batch conditions')
    old.save(out/'REPORT.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch-root',required=True);p.add_argument('--output',required=True);p.add_argument('--recovery-index');p.add_argument('--cohort-manifest')
    a=p.parse_args();print(json.dumps(audit_all(a.batch_root,a.output,a.recovery_index,a.cohort_manifest),indent=2))
