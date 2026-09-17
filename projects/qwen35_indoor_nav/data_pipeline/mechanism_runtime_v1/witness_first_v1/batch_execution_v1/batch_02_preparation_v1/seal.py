"""Read-only input verification and fresh batch02-only delivery manifest."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if p.name=='vla')


def sha(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    source=HERE/'composite/SOURCE_LOCK.json'
    lock=json.loads(source.read_text())
    for name,h in lock.items():assert sha(Path(name))==h,name
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    with (HERE/'SHA256SUMS').open('x') as f:
        for p in files:f.write(sha(p)+'  '+str(p.relative_to(HERE))+'\n')
    print(json.dumps(dict(verified_source_hashes=len(lock),delivery_files=len(files),
        source_lock_sha256=sha(source),config_draft_sha256=sha(HERE/'composite/CONFIG_DRAFT.json'))))


if __name__=='__main__':main()
