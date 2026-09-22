"""Bind the new implementation and read-only dependencies before first launch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
def main():
    assert not (u.HERE/'SOURCE_LOCK.json').exists(),'ALREADY_LOCKED'
    files={}
    for path,h in u.read(u.HERE.parent/'ordinary_stop_boundary_v14/SOURCE_LOCK.json')['files'].items():
        if '/ordinary_stop_boundary_v14/' in path:continue
        assert u.sha(Path(path))==h,'READ_ONLY_DEPENDENCY_CHANGED:'+path
        files[path]=h
    extra=[u.LINE/'data_pipeline/ordinary_route_teacher_v11/route.py',u.ASSET.parents[1]/'third_party/habitat-lab/habitat/tasks/nav/nav.py']
    p=u.read(u.HERE/'PROTOCOL.json')
    extra += [Path(p['train_gt']),Path(p['unseen_gt'])]
    extra += list(u.HERE.glob('*.py'))+[u.HERE/'PROTOCOL.json',u.HERE/'index.html']+list((u.HERE/'manifests').glob('*.json'))
    for path in extra:files[str(path)]=u.sha(path)
    u.write(u.HERE/'SOURCE_LOCK.json',dict(source_commit=p['source_commit'],files=files,scene_fingerprint='full file SHA once in CPU geometry preflight; SCENE_IDENTITY.json'),True)
    print('LOCKED',len(files))
if __name__=='__main__':main()
