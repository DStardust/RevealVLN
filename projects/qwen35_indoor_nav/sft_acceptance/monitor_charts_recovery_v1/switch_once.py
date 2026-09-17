"""One-time exact monitor-only switch requested by the user. Never signals trainers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
PY = ROOT / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
OLD = HERE.parent / 'monitor_charts_v3/server.py'
TRAIN = HERE.parent / 'ordinary_sync_recovery_v1'


def identity(pid):
    p = Path('/proc') / str(pid)
    return dict(pid=pid, starttime=int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
                cwd=str((p / 'cwd').resolve()),
                argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v])


assert not (HERE / 'SWITCH_BEFORE.json').exists(), 'ONE_SHOT_ALREADY_ATTEMPTED'
assert subprocess.run(['tmux', 'has-session', '-t', 'q35n_charts_recovery_v1'],
                      capture_output=True).returncode != 0
old = identity(951905)
assert old['starttime'] == 144086899 and old['cwd'] == str(ROOT)
assert old['argv'] == [str(PY), '-I', '-S', '-B', str(OLD)]
status = json.loads((TRAIN / 'formal/STATUS.json').read_text())
launcher = identity(status['launcher_pid'])
assert str(TRAIN / 'launcher.py') in launcher['argv']
protocol = json.loads((TRAIN / 'PROTOCOL_FILESTORE.json').read_text())
for name, digest in protocol['code_sha256'].items():
    assert hashlib.sha256((TRAIN / name).read_bytes()).hexdigest() == digest
before = dict(unix=time.time(), old_monitor=old, launcher=launcher, status=status)
(HERE / 'SWITCH_BEFORE.json').write_text(json.dumps(before, indent=2))
fd = os.pidfd_open(old['pid'])
assert identity(old['pid']) == old
signal.pidfd_send_signal(fd, signal.SIGTERM)
os.close(fd)
for _ in range(40):
    if not Path('/proc', str(old['pid'])).exists():
        break
    time.sleep(.1)
assert not Path('/proc', str(old['pid'])).exists(), 'OLD_MONITOR_NOT_EXITED'
subprocess.run(['tmux', 'new-session', '-d', '-s', 'q35n_charts_recovery_v1', '-c', str(ROOT),
                f'exec {PY} -I -S -B {HERE / "server.py"}'], check=True)
for _ in range(40):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18766/api/status', timeout=5) as r:
            current = json.load(r)
        assert current['monitor_version'] == 'recovery_v1'
        break
    except (OSError, AssertionError):
        time.sleep(.25)
else:
    raise RuntimeError('NEW_MONITOR_HEALTH_FAILED; old sealed monitor can be relaunched')
with urllib.request.urlopen('http://127.0.0.1:18766/', timeout=5) as r:
    assert '已接入修复后的续训' in r.read().decode()
try:
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18766/api/status', method='POST'), timeout=5)
    raise AssertionError('MUTATING_HTTP_ALLOWED')
except urllib.error.HTTPError as e:
    assert e.code == 405
assert identity(launcher['pid']) == launcher, 'TRAINER_IDENTITY_CHANGED'
result = dict(status='PASS', unix=time.time(), monitor_port=18766,
              trainer_identity_unchanged=True, frozen_training_code_verified=True,
              post_http_status=405, monitor_version=current['monitor_version'],
              updates=current['progress']['data']['cursor']['updates'],
              progress_age_seconds=current['progress']['age_seconds'])
(HERE / 'SWITCH_RESULT.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
