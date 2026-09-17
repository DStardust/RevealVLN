"""One-shot user-authorized graceful request to the verified training rank zero."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
TRAIN = LINE / 'sft_acceptance/ordinary_sync_recovery_v1'

def identity(pid):
    p = Path('/proc') / str(pid)
    s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime_ticks=int(s[19]), ppid=int(s[1]),
                uid=p.stat().st_uid, cwd=str((p / 'cwd').resolve()),
                argv=(p / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))

def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

def main():
    assert not (HERE / 'TRAIN_STOP_REQUEST.json').exists(), 'ONE_SHOT_ONLY'
    rank = identity(1151311)
    assert rank['ppid'] == 1151310 and rank['uid'] == 0
    assert rank['cwd'] == str(TRAIN)
    assert rank['argv'] == [str(LINE / '.envs/q35n_qwen_g2_v1/bin/python3'),
        '-B', '-u', str(TRAIN / 'train_filestore.py'), '--protocol',
        str(TRAIN / 'PROTOCOL_FILESTORE.json'), '--run-dir',
        str(TRAIN / 'formal/attempt_001'), '--resume',
        str(LINE / 'sft_acceptance/ordinary_baseline_v3/formal/run_0001/checkpoint_000030400.pt')]
    env = dict(item.split(b'=', 1) for item in Path('/proc/1151311/environ').read_bytes().split(b'\0') if b'=' in item)
    assert env[b'RANK'] == b'0' and env[b'WORLD_SIZE'] == b'3'
    code = TRAIN / 'train_filestore.py'
    assert hashlib.sha256(code.read_bytes()).hexdigest() == 'b56913d7d6f109c5a29b932653b3e31ef36021bfe36ce5b02eb65c973e814c2a'
    tree = [identity(p) for p in (1151146, 1151306, 1151310, 1151311, 1151312, 1151313)]
    protected = [identity(p) for p in (1353423, 1353510, 1390468)]
    save('TRAIN_STOP_REQUEST.json', dict(user_request='把训练停了开始扩产吧',
         time_unix=time.time(), signal='SIGTERM', target=rank, training_tree=tree,
         protected_processes=protected, behavior='rank-zero flag, distributed consensus, final checkpoint, normal supervisor closure, holder restoration',
         frozen_sources_modified=False, automatic_training_restart_allowed=False))
    assert identity(rank['pid']) == rank, 'IDENTITY_CHANGED_BEFORE_SIGNAL'
    os.kill(rank['pid'], signal.SIGTERM)
    save('TRAIN_STOP_SIGNAL_SENT.json', dict(time_unix=time.time(), pid=rank['pid']))
    print('GRACEFUL_STOP_REQUESTED_RANK_ZERO_ONLY', flush=True)

if __name__ == '__main__':
    main()
