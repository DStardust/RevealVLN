import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seal(snapshot):
    snapshot=snapshot.resolve()
    if snapshot.parent!=HERE or not snapshot.name.startswith('snapshot'):raise ValueError('SNAPSHOT_SCOPE')
    lock=json.loads((snapshot/'SOURCE_LOCK.json').read_text())
    for path,h in lock.items():assert sha(Path(path))==h,path
    files=sorted(p for p in snapshot.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    with (snapshot/'SHA256SUMS').open('x') as f:
        for path in files:f.write(sha(path)+'  '+path.name+'\n')
    print(json.dumps({'source_hashes_verified':len(lock),'snapshot_files_sealed':len(files)}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('snapshot',type=Path)
    seal(parser.parse_args().snapshot)
