"""Seal completed CPU review without modifying the reviewed source."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'queue.py'
EXPECTED='e606eab59bb654bb31583333cfa2367780f908be70356594f97c9eeb27422683'
if __name__=='__main__':
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
    names=('test_safety.py','REPORT_ZH.md','result.json','seal.py')
    records=[(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),name) for name in names]
    with (HERE/'SHA256SUMS').open('x') as stream:
        stream.write(''.join(f'{sha}  {name}\n' for sha,name in records))
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for sha,name in records)
    print(EXPECTED,len(records),'REVIEW_FILES_VERIFIED')
