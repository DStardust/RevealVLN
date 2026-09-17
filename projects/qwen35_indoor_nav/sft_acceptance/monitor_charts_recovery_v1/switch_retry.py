"""Retry monitor switch: project stdlib lacks pidfd_open; first attempt sent no signal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
before = json.loads((HERE / 'SWITCH_BEFORE.json').read_text())
assert not (HERE / 'SWITCH_RESULT.json').exists()


def identity(pid):
    p = Path('/proc') / str(pid)
    return dict(pid=pid, starttime=int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
                cwd=str((p / 'cwd').resolve()),
                argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v])


old = before['old_monitor']
launcher = before['launcher']
assert identity(launcher['pid']) == launcher
assert identity(old['pid']) == old
assert old['pid'] == 951905 and old['starttime'] == 144086899
os.kill(old['pid'], signal.SIGTERM)
for _ in range(40):
    if not Path('/proc', str(old['pid'])).exists():
        break
    time.sleep(.1)
assert not Path('/proc', str(old['pid'])).exists()
py = old['argv'][0]
subprocess.run(['tmux', 'new-session', '-d', '-s', 'q35n_charts_recovery_v1', '-c', old['cwd'],
                f'exec {py} -I -S -B {HERE / "server.py"}'], check=True)
for _ in range(40):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18766/api/status', timeout=5) as r:
            current = json.load(r)
        assert current['monitor_version'] == 'recovery_v1'
        break
    except (OSError, AssertionError):
        time.sleep(.25)
else:
    raise RuntimeError('MONITOR_HEALTH_FAILED')
with urllib.request.urlopen('http://127.0.0.1:18766/', timeout=5) as r:
    assert '已接入修复后的续训' in r.read().decode()
try:
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18766/api/status', method='POST'), timeout=5)
    raise AssertionError('MUTATING_HTTP_ALLOWED')
except urllib.error.HTTPError as e:
    assert e.code == 405
assert identity(launcher['pid']) == launcher
result = dict(status='PASS', unix=time.time(), monitor_port=18766,
              trainer_identity_unchanged=True, frozen_training_code_verified_before_switch=True,
              previous_attempt='pidfd_open unavailable; no signal sent; preserved switch_once.py and SWITCH_BEFORE.json',
              post_http_status=405, monitor_version=current['monitor_version'],
              updates=current['progress']['data']['cursor']['updates'],
              progress_age_seconds=current['progress']['age_seconds'])
(HERE / 'SWITCH_RESULT.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
