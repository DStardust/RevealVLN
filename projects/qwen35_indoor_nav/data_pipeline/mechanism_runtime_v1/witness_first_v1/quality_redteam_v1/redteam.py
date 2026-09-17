"""Bounded TEST_FIXTURE corruptions of a real accepted family; CPU only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ACCEPT=WF/'quality_cpu/batch_acceptance_v1'
PRIOR=ACCEPT/'v3_final_audit'
RUN=WF/'short_revisit_v3/run_v1'
EXPORT=RUN/'bundles/WF_SHORT_REVISIT_V3_004/export_v4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);assert path.is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
def load():
    locks={x.split(None,1)[1]:x.split(None,1)[0] for x in (ACCEPT/'SHA256SUMS').read_text().splitlines()}
    assert sha(ACCEPT/'acceptance.py')==locks['acceptance.py']
    spec=importlib.util.spec_from_file_location('redteam_frozen_acceptance',ACCEPT/'acceptance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def reseal_export(root,evidence):
    names=[r.split('  ',1)[1] for r in (EXPORT/'SHA256SUMS').read_text().splitlines()]
    (root/'SHA256SUMS').write_text(''.join(sha(root/n)+'  '+n+'\n' for n in names))
    for n in names+['SHA256SUMS']:evidence['sealed_files'][str(root/n)]=sha(root/n)
def export_fixture(name,base):
    folder=HERE/'TEST_FIXTURES'/name;folder.mkdir(parents=True,exist_ok=False)
    root=folder/'export_v4';shutil.copytree(EXPORT,root)
    evidence=copy.deepcopy(base);reseal_export(root,evidence)
    save(folder/'TEST_FIXTURE.json',{'TEST_FIXTURE':True,'not_physical_generation':True,
        'training_admission':False,'scientific_pass':False,'source_export':str(EXPORT),
        'content_blobs':'original paths read-only; no copied or fabricated RGB/semantic data'})
    return folder,root,evidence
def rewrite(path,value,jsonl=False):
    assert Path(path).is_relative_to(HERE)
    Path(path).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in value) if jsonl else json.dumps(value,indent=2))

def main():
    assert not (HERE/'result.json').exists(),'NO_OVERWRITE'
    started=time.monotonic();a=load();q=a.quality;base=read(PRIOR/'EVIDENCE.json')
    # This lock inventories originals, not a claim of externally timestamped trust.
    source={str(p):sha(p) for p in (ACCEPT/'acceptance.py',WF/'quality_cpu/quality.py',
        q.RUNTIME/'loader.py',q.RUNTIME/'core_bridge.py',PRIOR/'EVIDENCE.json',PRIOR/'FAMILY_REPORT.json')}
    source.update(base['sealed_files']);save(HERE/'SOURCE_LOCK.json',source)
    positive=q.audit_family(EXPORT,base)
    save(HERE/'REAL_POSITIVE_REAUDIT.json',positive)
    assert positive['quality_pass'] is True,positive
    inspected=a.inspect_run(RUN)
    save(HERE/'REAL_SOURCE_REAUDIT.json',{'source_binding_pass':True,'run_root':str(inspected[0]),
        'config_sha256':inspected[3]['config_sha256'],'scientific_pass':False})
    results=[]
    def trial(name,mutate,expected,clone=False):
        if clone:folder,root,evidence=export_fixture(name,base)
        else:
            folder=HERE/'TEST_FIXTURES'/name;folder.mkdir(parents=True,exist_ok=False)
            root=EXPORT;evidence=copy.deepcopy(base)
            save(folder/'TEST_FIXTURE.json',{'TEST_FIXTURE':True,'not_physical_generation':True,
                'training_admission':False,'scientific_pass':False})
        mutate(folder,root,evidence)
        if clone:reseal_export(root,evidence)
        save(folder/'EVIDENCE_TEST_FIXTURE.json',evidence)
        report=q.audit_family(root,evidence);save(folder/'REPORT.json',report)
        rejected=report['quality_pass'] is False and any(expected in x for x in report['errors'])
        results.append({'name':name,'expected_rejection':expected,'correctly_rejected':rejected,
            'quality_pass':report['quality_pass'],'errors':report['errors'],
            'fixture_reseals_are_test_only':clone,'scientific_pass':False})
    def label(folder,root,evidence):
        path=root/'SUPERVISION_ONLY.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['y']=1-rows[0]['y'];rows[0]['outcome']='pass' if rows[0]['y'] else 'fail'
        rewrite(path,rows,True)
    trial('01_flipped_label_with_self_consistent_outcome',label,'LABEL_RECOMPUTATION',True)
    def counts(folder,root,evidence):
        manifest=read(root/'MANIFEST.json');candidate=manifest['candidate']
        candidate['histories']['H_A'].append('L')
        candidate['candidate_hash']=q.digest({k:v for k,v in candidate.items() if k!='candidate_hash'})
        rewrite(root/'MANIFEST.json',manifest);path=folder/'CANDIDATE_TEST_FIXTURE.json';save(path,candidate)
        evidence['candidate']=str(path);evidence['sealed_files'][str(path)]=sha(path)
    trial('02_history_count_shortcut_rehashed',counts,'ACTION_COUNT_SHORTCUT',True)
    trial('03_missing_one_certification_trace',lambda f,r,e:e['replay_paths'].pop(),'REPLAY_PATH_COUNT')
    def policy(folder,root,evidence):
        cell=json.loads((root/'SUPERVISION_ONLY.jsonl').read_text().splitlines()[0])
        path=root/cell['full_policy_path'];rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['query']=cell['query'];rewrite(path,rows,True)
    trial('04_future_query_added_to_policy',policy,'POLICY_FIELDS',True)
    trial('05_unsealed_candidate',lambda f,r,e:e['sealed_files'].pop(e['candidate']),'UNSEALED_FILE:')
    trial('06_changed_candidate_hash',lambda f,r,e:e['sealed_files'].__setitem__(e['candidate'],'0'*64),'SEALED_HASH_MISMATCH')
    # Exercise the frozen production builder's source gate, not merely audit_family.
    folder=HERE/'TEST_FIXTURES/07_changed_locked_source';folder.mkdir(parents=True)
    save(folder/'TEST_FIXTURE.json',{'TEST_FIXTURE':True,'not_physical_generation':True,'scientific_pass':False})
    names=('EXECUTION_CONFIG.json','INPUT_LOCK.json','PROCESS.json','SUPERVISOR_RESULT.json',
           'result.json','BUDGET_FINAL.json','STORE_CLOSE_AUDIT.json')
    for name in names:shutil.copyfile(RUN/name,folder/name)
    lock=read(folder/'INPUT_LOCK.json');lock[str(folder/'EXECUTION_CONFIG.json')]=lock.pop(str(RUN/'EXECUTION_CONFIG.json'))
    target=next(p for p in lock if p.endswith('/shared.py'))
    lock[target]='0'*64;rewrite(folder/'INPUT_LOCK.json',lock)
    try:
        a.inspect_run(folder);error=None
    except (ValueError,KeyError,OSError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
    row={'name':'07_changed_locked_source','entrypoint':'acceptance.inspect_run',
         'expected_rejection':'LOCKED_SOURCE_CHANGED:',
         'correctly_rejected':bool(error and 'LOCKED_SOURCE_CHANGED:' in error),
         'error':error,'scientific_pass':False};save(folder/'REPORT.json',row);results.append(row)
    # All original files must still match the pre-test lock, including raw content.
    unchanged={p:sha(p)==value for p,value in source.items()}
    save(HERE/'SOURCE_UNCHANGED.json',{'all_unchanged':all(unchanged.values()),'files':unchanged})
    result={'status':'CPU_REDTEAM_COMPLETE','TEST_FIXTURE':True,'positive_reaudit_pass':positive['quality_pass'],
        'negative_variants':len(results),'correctly_rejected':sum(r['correctly_rejected'] for r in results),
        'results':results,'original_files_unchanged':all(unchanged.values()),
        'elapsed_seconds':time.monotonic()-started,'new_physical_families':0,'training_admission':False,
        'scientific_pass':False,'scope':'bounded fault injection; not proof of complete validator soundness'}
    save(HERE/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
