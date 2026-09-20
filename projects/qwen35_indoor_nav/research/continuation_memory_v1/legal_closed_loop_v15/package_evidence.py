"""Bundle exact existing logs for review; raw scene arrays and weights stay local."""
import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review as r


def main():
    measured=r.c.read(r.HERE/'CONTINUATION_REVIEW.json')
    assert measured['completed_rollouts']==216
    raw=r.LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15'
    roots=[p for p in raw.glob('raw_run_*') if p.is_dir()]
    roots += [p for pattern in ('features_run_*','train_run_*','continuation_run_*') for p in r.HERE.glob(pattern) if p.is_dir()]
    files=set()
    for run in roots:
        # Explicit useful log locations; never traverse cache/content/checkpoints.
        for pattern in ('*.json','*.jsonl','*.log','source/*.py','source/*.json',
                        'rollouts/*/*.json','rollouts/*/*.jsonl','V15_*/*.json'):
            files.update(p for p in run.glob(pattern) if p.is_file())
    target=r.HERE/'EVIDENCE_LOGS.tar.gz';manifest=[]
    with target.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0,compresslevel=6) as compressed,tarfile.open(fileobj=compressed,mode='w|') as archive:
        for path in sorted(files):
            assert not path.is_symlink() and path.resolve().is_relative_to(r.LINE)
            data=path.read_bytes();name=str(path.relative_to(r.LINE))
            record=tarfile.TarInfo(name);record.size=len(data);record.mode=0o644;record.mtime=0
            archive.addfile(record,io.BytesIO(data))
            manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(target,'r:gz') as archive:
        members=archive.getmembers();assert len(members)==len(manifest)
        for member,entry in zip(members,manifest):
            assert member.name==entry['path'] and member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==entry['sha256']
    r.c.write(r.HERE/'EVIDENCE_MANIFEST.json',dict(archive=target.name,archive_sha256=r.c.sha(target),
        compressed_bytes=target.stat().st_size,members=manifest,readback_verified=True,
        contains='Original byte-exact run metadata, policy/input/action logs, privileged trace summaries, training steps and source snapshots, including failed attempts.',
        not_in_archive='MP3D assets, RGB/semantic NPY arrays, frozen base, feature tensors, learned checkpoints, compiled kernel binaries. These remain local with SHA references.'),True)
    print(dict(files=len(manifest),compressed_bytes=target.stat().st_size))


if __name__=='__main__':main()
