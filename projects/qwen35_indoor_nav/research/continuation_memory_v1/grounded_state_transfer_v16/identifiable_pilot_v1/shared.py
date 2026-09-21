"""One explicit scope for the identifiable-data pilot; old evidence is read-only."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
V16=HERE.parent
sys.path.append(str(V16))
from v16_common import c, read, write, append, sha, digest, immutable, atomic_torch, load, LINE, ROOT, DATA_LINE
DATA=V16/'identifiable_data_v1/final_verification_001'
OLD=V16/'runs/v16_fitdev_train_001'

def verify_lock(lock):
    for path,expected in lock['files'].items():
        if sha(LINE/path)!=expected:raise ValueError('SOURCE_CHANGED:'+path)

def config(run):
    value=read(run/'PROTOCOL.json');verify_lock(read(run/'SOURCE_LOCK.json'));return value
