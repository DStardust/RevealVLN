from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import gate


if __name__=='__main__':
    assert not (HERE/'INPUT_LOCK.json').exists()
    paths=list(HERE.glob('*.py'))+[gate.RECOVERY/'INPUT_LOCK.json',HERE.parent/'runtime_v1/INPUT_LOCK.json',HERE.parent/'runtime_v1/merge.py']
    lock={str(p.relative_to(gate.ROOT)):gate.sha(p) for p in paths}
    with (HERE/'INPUT_LOCK.json').open('x') as f:__import__('json').dump(lock,f,indent=2)
    print(gate.sha(HERE/'INPUT_LOCK.json'))
