"""Freeze code and read-only dependencies; future V15 head is bound after training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
def main():
    assert not (u.HERE/'SOURCE_LOCK.json').exists(),'ALREADY_LOCKED'
    source=u.LINE/'sft_acceptance/ordinary_stop_coverage_v15';files={}
    for lock in [source/'SOURCE_LOCK.json',source/'runtime_r3/SOURCE_LOCK.json',u.LINE/'sft_acceptance/ordinary_action_repair_v16/SOURCE_LOCK.json']:
        for path,h in u.read(lock)['files'].items():
            assert u.sha(Path(path))==h,'READ_ONLY_DEPENDENCY_CHANGED:'+path
            files[path]=h
        files[str(lock)]=u.sha(lock)
    for path in list(u.HERE.glob('*.py'))+[u.HERE/'PROTOCOL.json',u.HERE/'EXPOSURE_AUDIT.json']:
        files[str(path)]=u.sha(path)
    p=u.read(u.HERE/'PROTOCOL.json');assert u.sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    files[p['checkpoint']]=p['checkpoint_sha256'];assert u.sha(Path(p['reference_checkpoint']))==p['reference_checkpoint_sha256'];files[p['reference_checkpoint']]=p['reference_checkpoint_sha256']
    parent=u.LINE/'research/continuation_memory_v1/evidence_state_policy_v1/expanded_longtrain_v1'
    for rel,h in u.read(parent/'SOURCE_LOCK.json')['files'].items():
        path=u.LINE/rel
        assert u.sha(path)==h,'MEMORY_SOURCE_CHANGED:'+str(path)
        files[str(path)]=h
    files[str(parent/'SOURCE_LOCK.json')]=u.sha(parent/'SOURCE_LOCK.json')
    assert u.sha(Path(p['memory_checkpoint']))==p['memory_checkpoint_sha256']
    files[p['memory_checkpoint']]=p['memory_checkpoint_sha256']
    u.write(u.HERE/'SOURCE_LOCK.json',dict(source_commit=p['source_commit'],files=files,reference_binding='A=V13; B=committed100000-step memory+originalbest4k, all preexisting parameters frozen'),True)
    print('LOCKED',len(files))
if __name__=='__main__':main()
