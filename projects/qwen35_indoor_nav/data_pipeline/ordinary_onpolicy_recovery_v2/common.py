"""Project-local stdlib IO and frozen source verification."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_r2r_adapt_v5'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,data,exclusive=False):
    path=Path(path);tmp=path if exclusive else path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x' if exclusive else 'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    if not exclusive:tmp.replace(path)
def append(path,data):
    with Path(path).open('a') as f:f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def verify_lock():
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for path,digest in lock['files'].items():assert sha(path)==digest,'SOURCE_CHANGED:'+path
