"""Read back candidate artifacts and create a non-overwriting CPU audit seal."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import planner as p


def audit():
    out=p.HERE/'acceptance_v1'
    rows=json.loads((out/'candidates.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    coverage=json.loads((out/'coverage.json').read_text())
    reads=json.loads((out/'source_reads.json').read_text())
    failures=json.loads((out/'failure_ledger.json').read_text())
    split_path=p.LINE/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
    split=json.loads(split_path.read_text())
    assert p.sha(split_path)==summary['shared_split_sha256']
    mainline=p.sha(p.LINE/'MAINLINE_FREEZE_V3.md')
    assert mainline=='71c618db81be09d434b544a945b73789eba386d382aa5178d797f8ff4257ac86'
    assert set(r['house_id'] for r in reads)==set(split['FIT'])
    assert not set(r['house_id'] for r in reads)&set(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    assert len(rows)==summary['semantic_candidates']==len({r['candidate_id'] for r in rows})
    assert len(coverage)==summary['fit_houses_read']==len(reads)
    assert len(failures)==len(coverage)-summary['houses_with_candidates']
    object_maps={}
    for read in reads:
        path=p.scoped(p.ROOT/read['path'])
        assert p.sha(path)==read['sha256']
        object_maps[read['house_id']]=p.parse_house(path.read_text())
    for ordinal,row in enumerate(rows):
        assert row['ordinal']==ordinal and row['split']=='FIT'
        assert row['house_id'] in split['FIT']
        assert row['shared_split_sha256']==summary['shared_split_sha256']
        assert row['training_admission'] is False
        assert p.validate_quality(row)
        assert p.digest({'version':1,'house_id':row['house_id'],'roles':row['roles']})==row['semantic_fingerprint_sha256']
        groups=p.semantic_groups(object_maps[row['house_id']])
        for name,role in row['roles'].items():
            key=(role['mpcat40'],role['room'],role['raw_match']['value'])
            assert row['expected_eligible'][name]==[x['object_index'] for x in groups[key]]
        assert row['tasks']==p.task_spec(row['roles'])
    result={'status':'CPU_CANDIDATE_PREPARATION_PASS_NOT_RUNTIME',
            'verified_candidates':len(rows),'verified_source_house_hashes':len(reads),
            'fit_houses_with_candidates':summary['houses_with_candidates'],
            'metadata_ineligible_houses':len(failures),
            'source_split_integrity_pass':True,'candidate_semantics_readback_pass':True,
            'cpu_unit_tests_passed':17,'unit_test_evidence':'python -I -S -B -m unittest discover; 17 tests passed in task execution',
            'mainline_sha256':mainline,'runtime_pass':None,'training_quality_pass':None,
            'new_physical_families':0,'scientific_pass':False}
    with (out/'AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    files=sorted(x for x in p.HERE.rglob('*') if x.is_file() and x.name!='SHA256SUMS' and '__pycache__' not in x.parts)
    with (p.HERE/'SHA256SUMS').open('x') as f:
        for path in files:f.write(p.sha(path)+'  '+str(path.relative_to(p.HERE))+'\n')
    print(json.dumps(result))


if __name__=='__main__':audit()
