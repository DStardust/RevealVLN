"""Preview, verify exact old process, replace only original training monitor, rollback."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import urllib.error

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
OLD=HERE.parent/'monitor_charts_recovery_v1/server.py'
PANE='%294'
def ident(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,start=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),
                cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1])
def tmux(*args):return subprocess.check_output(['tmux',*args],text=True,timeout=10).strip()
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,indent=2)
def fetch(port,path='/api/status'):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=15) as f:return f.read()


def check(port):
    d=json.loads(fetch(port));assert d['monitor_version']=='ordinary_expanded_v1'
    assert all(p.get('segment')!='legacy_v3' for p in d['points'])
    assert '51,301' in fetch(port,'/').decode()
    try:
        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/status',method='POST'),timeout=10)
        raise AssertionError('POST_ALLOWED')
    except urllib.error.HTTPError as e:assert e.code==405
    return d


def main():
    old=ident(1201250)
    assert old['argv']==[str(PY),'-I','-S','-B',str(OLD)] and old['cwd']==str(ROOT)
    assert tmux('display-message','-p','-t',PANE,'#{pane_pid}')==str(old['pid'])
    protected={pid:ident(pid) for pid in (1731238,1570518,1564204,3996270,112240,1563749,1353421)}
    with (HERE/'preview.log').open('x') as log:
        preview=subprocess.Popen([str(PY),'-I','-S','-B',str(HERE/'server.py'),'--port','18770'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(40):
                assert preview.poll() is None,'PREVIEW_FAILED'
                try:d=check(18770);break
                except OSError:time.sleep(.25)
            else:raise RuntimeError('PREVIEW_TIMEOUT')
        finally:
            if preview.poll() is None:preview.terminate()
            preview.wait(timeout=10)
    save('DEPLOY_BEFORE.json',dict(old_monitor=old,protected=protected,preview_version=d['monitor_version']))
    tmux('set-option','-w','-t',PANE,'remain-on-exit','on')
    assert ident(old['pid'])==old
    fd=os.pidfd_open(old['pid']);assert ident(old['pid'])==old
    signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
    try:
        for _ in range(80):
            if tmux('display-message','-p','-t',PANE,'#{pane_dead}')=='1':break
            time.sleep(.1)
        else:raise RuntimeError('OLD_MONITOR_NOT_EXITED')
        tmux('respawn-pane','-t',PANE,'-c',str(ROOT),f'exec {PY} -I -S -B {HERE/"server.py"}')
        for _ in range(40):
            try:d=check(18766);break
            except OSError:time.sleep(.25)
        else:raise RuntimeError('DEPLOY_HEALTH_TIMEOUT')
        assert all(ident(pid)==row for pid,row in protected.items()),'PROTECTED_PROCESS_CHANGED'
        save('DEPLOY_RESULT.json',dict(status='PASS',unix=time.time(),new_monitor=ident(int(tmux('display-message','-p','-t',PANE,'#{pane_pid}'))),
             port=18766,version=d['monitor_version'],protected_identities_unchanged=True,only_old_monitor_signalled=True,post_status=405))
    except BaseException:
        # Roll back only an identified replacement belonging to this pane.
        pid=int(tmux('display-message','-p','-t',PANE,'#{pane_pid}'))
        if Path('/proc',str(pid)).exists():
            now=ident(pid)
            assert str(HERE/'server.py') in now['argv'],'ROLLBACK_UNKNOWN_PROCESS'
            fd=os.pidfd_open(pid);assert ident(pid)==now;signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
            for _ in range(80):
                if tmux('display-message','-p','-t',PANE,'#{pane_dead}')=='1':break
                time.sleep(.1)
        tmux('respawn-pane','-t',PANE,'-c',str(ROOT),f'exec {PY} -I -S -B {OLD}')
        save('ROLLBACK.json',dict(unix=time.time(),old_monitor_restored=True));raise
    print(json.dumps(dict(status='PASS',port=18766)))


if __name__=='__main__':main()
