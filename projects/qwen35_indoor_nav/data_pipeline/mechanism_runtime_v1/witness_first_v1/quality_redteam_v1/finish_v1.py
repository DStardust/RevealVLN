"""Complete preserved six-case run after an absent shared.py fixture assumption."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('redteam_cpu',Path(__file__).with_name('redteam.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def main():
    a=r.load();folder=r.HERE/'TEST_FIXTURES/07_changed_locked_source'
    r.save(r.HERE/'HARNESS_INTERRUPTION.json',{'stage':'seventh_fixture_preparation',
        'exception':'StopIteration selecting /shared.py from original V3 source lock',
        'cause':'V3 predates batch shared.py; fixture assumption, not validator failure',
        'first_six_reports_preserved':True,'seventh_validator_not_previously_called':True,
        'correction':'select existing sealed core_bridge.py; original sources and six outcomes unchanged'})
    lock=r.read(folder/'INPUT_LOCK.json')
    lock[str(folder/'EXECUTION_CONFIG.json')]=lock.pop(str(r.RUN/'EXECUTION_CONFIG.json'))
    target=next(p for p in lock if p.endswith('/core_bridge.py'))
    lock[target]='0'*64;r.rewrite(folder/'INPUT_LOCK.json',lock)
    try:a.inspect_run(folder);error=None
    except (ValueError,KeyError,OSError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
    row={'name':'07_changed_locked_source','entrypoint':'acceptance.inspect_run',
        'expected_rejection':'LOCKED_SOURCE_CHANGED:',
        'correctly_rejected':bool(error and 'LOCKED_SOURCE_CHANGED:' in error),
        'error':error,'scientific_pass':False}
    r.save(folder/'REPORT.json',row)
    expected=['LABEL_RECOMPUTATION','ACTION_COUNT_SHORTCUT','REPLAY_PATH_COUNT',
              'POLICY_FIELDS','UNSEALED_FILE:','SEALED_HASH_MISMATCH']
    reports=[]
    folders=sorted((r.HERE/'TEST_FIXTURES').iterdir())
    for folder,code in zip(folders[:6],expected):
        report=r.read(folder/'REPORT.json')
        reports.append({'name':folder.name,'expected_rejection':code,
            'correctly_rejected':report['quality_pass'] is False and any(code in e for e in report['errors']),
            'quality_pass':report['quality_pass'],'errors':report['errors'],'scientific_pass':False})
    reports.append(row)
    source=r.read(r.HERE/'SOURCE_LOCK.json');unchanged={p:r.sha(p)==value for p,value in source.items()}
    r.save(r.HERE/'SOURCE_UNCHANGED.json',{'all_unchanged':all(unchanged.values()),'files':unchanged})
    result={'status':'CPU_REDTEAM_COMPLETE','TEST_FIXTURE':True,
        'positive_reaudit_pass':r.read(r.HERE/'REAL_POSITIVE_REAUDIT.json')['quality_pass'],
        'source_builder_positive_pass':r.read(r.HERE/'REAL_SOURCE_REAUDIT.json')['source_binding_pass'],
        'negative_variants':7,'correctly_rejected':sum(x['correctly_rejected'] for x in reports),
        'results':reports,'original_files_unchanged':all(unchanged.values()),
        'original_files_verified':len(source),'new_physical_families':0,'training_admission':False,
        'scientific_pass':False,'harness_interruption_preserved':'HARNESS_INTERRUPTION.json',
        'scope':'bounded fault injection; not proof of complete validator soundness'}
    r.save(r.HERE/'result.json',result)
    print(r.json.dumps(result,indent=2))

if __name__=='__main__':main()
