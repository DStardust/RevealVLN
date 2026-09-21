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

PILOT=V16/'identifiable_pilot_v1'
PRIOR_RUN=PILOT/'runs/pilot_001'
COVERAGE=V16/'b2_state_coverage_v1/runs/coverage_001'

def runtime_config(run):
    import os
    value=read(run/'PROTOCOL.json')
    if 'B2_DEVICE' in os.environ:value.update(read(Path(os.environ['B2_DEVICE'])))
    return value

CONTROL=V16/'b2_matched_control_v1/runs/matched_001'
