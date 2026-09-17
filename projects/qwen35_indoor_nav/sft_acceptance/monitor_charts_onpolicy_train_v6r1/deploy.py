"""Health-check first, replace exactly one monitor, preserve and rollback if needed."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
OLD=HERE.parent/'monitor_charts_onpolicy_train_v6/server.py';NEW=HERE/'server.py'
PANE='%294';OLD_PID=2894388


def ident(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,start=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),
                cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1])
def tmux(*args):return subprocess.check_output(['tmux',*args],text=True,timeout=10).strip()
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def fetch(port,path='/api/status'):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=15) as f:return f.read()


def check(port):
    d=json.loads(fetch(port))
    assert d['monitor_version']=='ordinary_onpolicy_train_v6r1'
    assert d['snapshot_counts']['instruction_conditioned_decisions']==2650347
    assert d['training_completion']['completed_budget'] and d['training_completion']['holder_restoration_verified']
    assert d['navigation']['before']['result']['sr']==.14 and d['navigation']['matched']['result']['sr']==.21
    assert 'navigation-results' in fetch(port,'/').decode()
    for method in ('POST','PUT','DELETE'):
        try:
            urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/status',method=method),timeout=10)
            raise AssertionError('WRITE_METHOD_ALLOWED')
        except urllib.error.HTTPError as e:assert e.code==405
    return d


def wait_dead():
    for _ in range(80):
        if tmux('display-message','-p','-t',PANE,'#{pane_dead}')=='1':return
        time.sleep(.1)
    raise RuntimeError('MONITOR_NOT_EXITED')


def protected_pids():
    import xml.etree.ElementTree as ET
    pids={3996270,112240,1563749,1353421}
    data=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    # All GPU identities are read-only protected, including foreign GPU0.
    for gpu in data.findall('gpu'):
        for row in gpu.findall('processes/process_info'):pids.add(int(row.findtext('pid')))
    return tuple(pids)

def main():
    old=ident(OLD_PID)
    assert old['argv']==[str(PY),'-I','-S','-B',str(OLD)] and old['cwd']==str(ROOT)
    assert tmux('display-message','-p','-t',PANE,'#{pane_pid}')==str(old['pid'])
    old_hash=hashlib.sha256(OLD.read_bytes()).hexdigest()
    protected={pid:ident(pid) for pid in protected_pids()}
    test=subprocess.run([str(PY),'-I','-S','-B',str(HERE/'tests.py')],cwd=ROOT,capture_output=True,text=True,timeout=30)
    save('CPU_TEST_RESULT.json',dict(passed=test.returncode==0,returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,browser_visual_test_performed=False))
    assert test.returncode==0,test.stderr
    with (HERE/'preview.log').open('x') as log:
        preview=subprocess.Popen([str(PY),'-I','-S','-B',str(NEW),'--port','18770'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(40):
                assert preview.poll() is None,'PREVIEW_FAILED'
                try:d=check(18770);break
                except OSError:time.sleep(.25)
            else:raise RuntimeError('PREVIEW_TIMEOUT')
        finally:
            if preview.poll() is None:preview.terminate()
            preview.wait(timeout=10)
    save('DEPLOY_BEFORE.json',dict(old_monitor=old,old_sha256=old_hash,protected=protected,preview_passed=True))
    tmux('set-option','-w','-t',PANE,'remain-on-exit','on')
    assert ident(old['pid'])==old
    assert tmux('display-message','-p','-t',PANE,'#{pane_pid}')==str(old['pid'])
    assert ident(old['pid'])==old
    os.kill(old['pid'],signal.SIGTERM)
    try:
        wait_dead()
        tmux('respawn-pane','-t',PANE,'-c',str(ROOT),f'exec {PY} -I -S -B {NEW}')
        for _ in range(40):
            try:d=check(18766);break
            except OSError:time.sleep(.25)
        else:raise RuntimeError('DEPLOY_HEALTH_TIMEOUT')
        assert all(ident(pid)==row for pid,row in protected.items()),'PROTECTED_IDENTITY_CHANGED'
        assert hashlib.sha256(OLD.read_bytes()).hexdigest()==old_hash
        save('DEPLOY_RESULT.json',dict(status='PASS',unix=time.time(),new_monitor=ident(int(tmux('display-message','-p','-t',PANE,'#{pane_pid}'))),
            port=18766,protected_identities_unchanged=True,old_source_unchanged=True,only_old_monitor_signaled=True,write_methods_status=405))
    except BaseException:
        if tmux('display-message','-p','-t',PANE,'#{pane_dead}')!='1':
            pid=int(tmux('display-message','-p','-t',PANE,'#{pane_pid}'));now=ident(pid)
            assert now['argv']==[str(PY),'-I','-S','-B',str(NEW)] and now['cwd']==str(ROOT),'ROLLBACK_UNKNOWN_PROCESS'
            assert ident(pid)==now;os.kill(pid,signal.SIGTERM);wait_dead()
        tmux('respawn-pane','-t',PANE,'-c',str(ROOT),f'exec {PY} -I -S -B {OLD}')
        save('ROLLBACK.json',dict(unix=time.time(),old_monitor_restored=True));raise
    print(json.dumps(dict(status='PASS',port=18766)))


if __name__=='__main__':main()
