"""Run sealed control regressions with explicit FileStore; no TCP rendezvous daemon."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent


def main():
    if '--rank-worker' in sys.argv:
        sys.path.insert(0, str(HERE))
        import torch
        import test_control
        original = torch.distributed.init_process_group

        def init(backend, **kwargs):
            return original(backend, init_method=Path(os.environ['Q35N_STORE']).as_uri(),
                            rank=int(os.environ['RANK']), world_size=3, **kwargs)

        torch.distributed.init_process_group = init
        test_control.distributed_test()
        return
    out = HERE / 'cpu_filestore_v1'
    out.mkdir(exist_ok=False)
    workers = []
    logs = []
    try:
        for rank in range(3):
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='3',
                       CUDA_VISIBLE_DEVICES='', GLOO_SOCKET_IFNAME='lo', OMP_NUM_THREADS='1',
                       Q35N_STORE=str(out / 'rendezvous'))
            log = (out / ('rank_%d.log' % rank)).open('xb')
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__)),
                                              '--rank-worker'], env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 90
        while any(p.poll() is None for p in workers) and time.monotonic() < deadline:
            time.sleep(.25)
        codes = [p.poll() for p in workers]
        assert codes == [0, 0, 0], codes
        result = dict(status='PASS', exit_codes=codes, backend='gloo', store='file',
                      tests='THREE_RANK_CLOCK_SKEW_SINGLE_RANK_STOP_AND_WALL_BUDGET')
        with (out / 'RESULT.json').open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
