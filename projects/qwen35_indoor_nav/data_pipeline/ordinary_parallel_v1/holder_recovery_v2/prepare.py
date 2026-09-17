"""Freeze CPU recovery code; no process reads, GPU queries or signals."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import recover


def main():
    assert not (HERE/'INPUT_LOCK.json').exists()
    paths=list(HERE.glob('*.py'))
    for gpu in recover.EXPECTED_RUNTIME_LOCKS:
        runtime=recover.runtime_path(gpu)
        paths.append(runtime/'INPUT_LOCK.json')
        paths.append(runtime/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')
    lock={str(p.relative_to(recover.ROOT)):recover.sha(p) for p in paths}
    recover.save(HERE/'INPUT_LOCK.json',lock)
    print(__import__('json').dumps({str(g):recover.approval_value(g) for g in recover.EXPECTED_RUNTIME_LOCKS},indent=2))


if __name__=='__main__':main()
