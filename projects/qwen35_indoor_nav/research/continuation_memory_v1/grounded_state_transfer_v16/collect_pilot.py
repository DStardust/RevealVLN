"""Independent FIT/DEV physical pilot preceding TEST condition freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from pipeline import execute, preflight

def main(run,previous=None):
    run.mkdir(parents=True,exist_ok=False)
    config=read(HERE/'PROTOCOL.json');immutable(run/'PROTOCOL.json',config)
    paths=[HERE/'collect.py',HERE/'evaluator_v16.py',HERE/'v16_common.py',HERE/'build_manifest.py']
    immutable(run/'COLLECTION_SOURCE_LOCK.json',dict(baseline_commit=BASELINE,files={str(p.relative_to(LINE)):sha(p) for p in paths}))
    # Pilot has no dependency on future training implementation or any TEST score.
    immutable(run/'SOURCE_LOCK.json',source_lock())
    if previous:
        old=Path(previous)
        records={p.name.removeprefix('PRIOR_'):p for p in old.glob('PRIOR_HOUSE_*.json')}
        records.update({p.name:p for p in old.glob('HOUSE_*.json')})
        for name,path in records.items():
            record=read(path)
            immutable(run/(name if record['complete'] else 'PRIOR_'+name),record)
        immutable(run/'REUSED_PHYSICAL_ASSETS.json',dict(source=str(old),houses=[p.name for p in old.glob('HOUSE_*.json')],
            reuse='Completed families remain read-only. Incomplete houses may receive only new preregistered physical proposals; previous proposals are excluded and attempts retained.',
            previous_protocol_sha256=sha(old/'PROTOCOL.json'),previous_source_lock_sha256=sha(old/'SOURCE_LOCK.json')))
    preflight(run,config)
    try:execute(run,config,'collect_fit_dev',[HERE/'collect.py',run,'FIT','DEV'],LINE/'.envs/q35n_habitat_v017_g0r/bin/python3')
    except BaseException as exc:
        write(run/'PILOT_RESULT.json',dict(status='STOPPED',reason=repr(exc),test_started=False),True);raise
    write(run/'PILOT_RESULT.json',dict(status='FIT_DEV_PHYSICAL_COLLECTION_COMPLETE',test_started=False),True)

if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else None)
