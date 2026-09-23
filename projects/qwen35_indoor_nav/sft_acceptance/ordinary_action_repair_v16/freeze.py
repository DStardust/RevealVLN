"""Freeze code and read-only dependencies; future V15 head is bound after training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
def main():
    assert not (u.HERE/'SOURCE_LOCK.json').exists(),'ALREADY_LOCKED'
    source=u.HERE.parent/'ordinary_stop_coverage_v15';files={}
    for lock in [source/'SOURCE_LOCK.json',source/'runtime_r3/SOURCE_LOCK.json']:
        for path,h in u.read(lock)['files'].items():
            assert u.sha(Path(path))==h,'READ_ONLY_DEPENDENCY_CHANGED:'+path
            files[path]=h
        files[str(lock)]=u.sha(lock)
    for path in list(u.HERE.glob('*.py'))+[u.HERE/'PROTOCOL.json',u.HERE/'index.html']:
        files[str(path)]=u.sha(path)
    p=u.read(u.HERE/'PROTOCOL.json');assert u.sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    files[p['checkpoint']]=p['checkpoint_sha256']
    u.write(u.HERE/'SOURCE_LOCK.json',dict(source_commit=p['source_commit'],files=files,reference_binding='V15 actual TRAIN_RESULT checkpoint SHA at TRAIN; verify again at every evaluation session; no substitute reference'),True)
    print('LOCKED',len(files))
if __name__=='__main__':main()
