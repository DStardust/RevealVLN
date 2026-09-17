"""Bounded single-node three-rank launcher with a fresh file rendezvous per attempt."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    assert Path(args[0]).resolve() == HERE / 'train_filestore.py'
    run = Path(args[args.index('--run-dir') + 1]).resolve()
    assert run.is_relative_to(HERE / 'formal')
    rendezvous = run / 'rendezvous_file'
    assert not rendezvous.exists(), 'FRESH_STORE_REQUIRED'
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _: stopped.append(s))
    children = []
    try:
        for rank in range(3):
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='3',
                       Q35N_STORE=str(rendezvous), OMP_NUM_THREADS='1')
            children.append(subprocess.Popen([sys.executable, '-B', '-u', *args], env=env))
        while not stopped:
            codes = [p.poll() for p in children]
            if all(c is not None for c in codes):
                return 0 if codes == [0, 0, 0] else 1
            if any(c not in (None, 0) for c in codes):
                return 1
            time.sleep(.5)
        return 1
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + 15
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(.5)
        for p in children:
            if p.poll() is None:
                p.kill()
            p.wait()


if __name__ == '__main__':
    sys.exit(main())
