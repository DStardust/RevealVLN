"""Verify the publication from its archived bytes without model/GPU access."""
import csv, hashlib, json, math, tarfile
from pathlib import Path


def main(root):
    read=lambda name:json.loads((root/name).read_text())
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    snapshot=read('SNAPSHOT.json')
    entries=read('TRACE_INDEX.json');expected={x['path']:x for x in entries}
    assert len(expected)==len(entries),'DUPLICATE_ARCHIVE_PATH'
    observed=set();pairs={}
    for archive in read('TRACE_ARCHIVES.json'):
        path=root/archive['path'];assert path.stat().st_size==archive['bytes']
        assert sha(path.read_bytes())==archive['sha256']
        assert archive['bytes']<100_000_000,'GITHUB_FILE_SIZE_LIMIT'
        with tarfile.open(path,'r|gz') as tar:
            for entry in tar:
                assert entry.isfile() and entry.name in expected and entry.name not in observed
                spec=expected[entry.name];raw=tar.extractfile(entry).read()
                assert len(raw)==spec['bytes'] and sha(raw)==spec['sha256']
                observed.add(entry.name)
                if spec['kind']=='PAIR_SEAL':pairs[entry.name]=json.loads(raw)
    assert observed==set(expected),'MISSING_SEALED_TRACE'
    captured=0
    for item in read('CAPTURE_INDEX.json'):
        if 'snapshot_path' not in item:continue
        assert sha((root/item['snapshot_path']).read_bytes())==item['sha256'];captured+=1
    with (root/'ROLLOUTS.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==snapshot['planned_pair_slots']
    complete=[x for x in rows if x['status']=='COMPLETE_PAIR_SEALED']
    assert len(complete)==len(pairs)==snapshot['complete_pairs']
    for row in complete:
        value=pairs[row['source_path']]
        assert value['rank']==int(row['rank']) and value['episode_id']==int(row['episode_id'])
        for arm in ('A','B'):
            result=value[arm]
            assert result['success']==float(row[arm+'_success'])
            assert result['success']==float(result['stopped'] and result['distances'][-1]<3)
            assert 0<result['steps']<=500
            assert len(result['distances'])==len(result['positions'])==result['steps']+1
            assert result['stopped'] or result['steps']==500
            assert all(math.isfinite(result[key]) for key in ('success','spl','ndtw','oracle_success'))
    for summary in snapshot['summaries']:
        group=[x for x in rows if x['experiment']==summary['experiment'] and x['phase']==summary['phase']]
        done=[x for x in group if x['status']=='COMPLETE_PAIR_SEALED']
        assert len(group)==summary['planned_pairs'] and len(done)==summary['complete_pairs']
        for arm in ('A','B'):assert sum(float(x[arm+'_success']) for x in done)==summary[arm]['successes']
    result=dict(status='PUBLICATION_BYTES_AND_COUNTS_VERIFIED',archive_files=len(read('TRACE_ARCHIVES.json')),sealed_trace_files=len(observed),captured_files=captured,complete_pairs=len(pairs),planned_pair_slots=len(rows),checks=['Archive and individual trace SHA256','Captured metadata SHA256','Unique immutable seals and full pending denominator','CSV/archived pair aggregate agreement','Active STOP with final distance<3m; <=500 steps; complete trajectory arrays'],scope='CPU publication integrity only; no new model run, no re-evaluation of model-input tensors or geodesic backend')
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    main(Path(__file__).resolve().parent)
