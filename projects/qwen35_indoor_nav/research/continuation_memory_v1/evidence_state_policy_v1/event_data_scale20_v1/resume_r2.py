"""Preserve interrupted array writes and seal accepted evidence before resuming."""
import os
import re
import time
from pathlib import Path
from shared import read,write,sha,immutable,append,LINE,HERE

def quarantine_array_temporaries(run,house):
    run=Path(run).resolve()
    if not run.is_relative_to((HERE/'runs').resolve()):raise ValueError('RUN_BOUNDARY')
    content=run/'collect'/house/'content'
    if not content.exists():return []
    if content.is_symlink():raise ValueError('CONTENT_SYMLINK')
    records=[]
    # Only generated temporary array names are eligible. Never alter complete arrays.
    for name in os.listdir(content):
        if not re.fullmatch(r'[0-9a-f]{64}\.(rgb|semantic)\.npy\.tmp',name):continue
        path=content/name
        if path.is_symlink() or path.stat().st_uid!=os.getuid():raise ValueError('TEMP_OWNERSHIP')
        folder=run/'interrupted_writes_r2'/house;folder.mkdir(parents=True,exist_ok=True)
        target=folder/(name+'.'+str(time.time_ns()))
        record=dict(source=str(path.relative_to(run)),retained_as=str(target.relative_to(run)),
                    sha256=sha(path),bytes=path.stat().st_size,reason='Interrupted own array write; never promoted to complete data')
        os.rename(path,target)
        if sha(target)!=record['sha256']:raise ValueError('QUARANTINE_HASH_CHANGED')
        append(run/'INTERRUPTED_WRITES_R2.jsonl',record);records.append(record)
    return records

def preserve_complete_metadata(run):
    files={}
    for path in sorted((run/'collect').glob('*/*/FAMILY.json')):
        files[str(path.relative_to(run))]=sha(path)
        family=read(path);audit=LINE/family['audit']['path']
        if sha(audit)!=family['audit']['sha256']:raise ValueError('PRIOR_AUDIT_CHANGED')
        files[str(audit.relative_to(run))]=sha(audit)
    previous=run/'PRESERVED_METADATA_R2.json'
    if previous.exists():
        for p,expected in read(previous)['files'].items():
            if sha(run/p)!=expected:raise ValueError('COMPLETE_FAMILY_CHANGED:'+p)
    else:immutable(previous,dict(files=files,variants=len(files)//2,scope='Original FAMILY and AUDIT metadata; old full array audits retained'))
    return len(files)//2

