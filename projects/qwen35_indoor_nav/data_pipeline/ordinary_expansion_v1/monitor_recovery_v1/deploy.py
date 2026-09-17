"""One-shot exact monitor PID switch; no production/GPU process controls."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import monitor as m
PY=str(m.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3')
OLD_ARGV=[PY,'-I','-S','-B',str(m.OLD/'monitor.py')]
NEW_ARGV=[PY,'-I','-S','-B',str(HERE/'monitor.py')]
PANE='%302'

def save(name,value):
    with (HERE/name).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())

def call(*args):
    return subprocess.check_output(args,text=True,timeout=15).strip()

def identity(pid):
    p=Path('/proc')/str(pid);stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime_ticks=int(stat[19]),uid=p.stat().st_uid,
        cwd=str((p/'cwd').resolve(strict=True)),argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))

def pane(fmt):
    return call('tmux','display-message','-p','-t',PANE,fmt)

def wait_dead():
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        assert pane('#{pane_id}')==PANE
        if pane('#{pane_dead}')=='1':return
        time.sleep(.2)
    raise RuntimeError('MONITOR_PANE_NOT_DEAD_NO_ESCALATION')

def readiness(port,version):
    deadline=time.monotonic()+30;error=None
    while time.monotonic()<deadline:
        try:
            base=f'http://127.0.0.1:{port}'
            with urllib.request.urlopen(base+'/healthz',timeout=2) as r:health=json.load(r)
            if version:
                assert health['version']==m.VERSION and health['data_ready'], 'NEW_MONITOR_NOT_READY'
            with urllib.request.urlopen(base+'/api/status',timeout=2) as r:state=json.load(r)
            assert len(state['lanes'])==3
            if version:assert state['monitor']['version']==m.VERSION and not state['monitor']['stale']
            with urllib.request.urlopen(base+'/?reload=1' if version else base+'/',timeout=2) as r:page=r.read().decode()
            assert '普通导航数据扩产' in page
            if version:assert '最近成功采集' in page and 'createPoller' in page
            return dict(health=health,completed=state['completed'],strict_routes=state['strict_routes'],
                        stages=[dict(gpu=l['gpu'],stage=l['stage']) for l in state['lanes']])
        except Exception as exc:error=repr(exc);time.sleep(.25)
    raise RuntimeError('MONITOR_READINESS_TIMEOUT:'+str(error))

def main():
    assert not (HERE/'DEPLOY_BEFORE.json').exists(),'ONE_SHOT_NODE'
    cpu=m.read(HERE/'CPU_TESTS.json');js=m.read(HERE/'CLIENT_TESTS.json')
    assert cpu['passed'] and cpu['tests']==8 and js['passed'] and js['tests']==8
    for report,field in ((cpu,'tested_sha256'),(js,'source_sha256')):
        for name,digest in report[field].items():assert m.sha(HERE/name)==digest,('TESTED_SOURCE_CHANGED',name)
    m.load_collector()  # Validate old input lock and monitor source, without running a GPU.
    old=identity(1421413)
    assert old['argv']==OLD_ARGV and old['cwd']==str(m.ROOT) and old['uid']==0
    assert int(pane('#{pane_pid}'))==old['pid'] and pane('#{pane_dead}')=='0'
    assert 'pid=1421413,' in call('ss','-ltnpH','sport = :18769')
    protected=[identity(p) for p in (1421392,1421592,1421593)]
    remain=call('tmux','show-options','-w','-v','-t',PANE,'remain-on-exit')
    inputs={str(p.relative_to(m.ROOT)):m.sha(p) for p in (m.OLD/'INPUT_LOCK.json',m.OLD/'monitor.py',
        m.LINE/'sft_acceptance/ordinary_sync_recovery_v1/PROTOCOL_FILESTORE.json',
        m.LINE/'closed_loop_bench/r2r_ce_full_v2/PROTOCOL.json')}
    sources={p.name:m.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.js','.html','.md')}
    save('MONITOR_FREEZE.json',dict(source_sha256=sources,old_input_hashes=inputs,production_mutation_allowed=False))
    save('DEPLOY_BEFORE.json',dict(old_monitor=old,pane=PANE,remain_on_exit=remain,protected=protected,
        frozen_inputs=inputs,time_unix=time.time(),new_argv=NEW_ARGV))
    preview=None;new=None;switched=False
    try:
        with (HERE/'PREVIEW.log').open('x') as log:
            preview=subprocess.Popen(NEW_ARGV+['--port','18770'],cwd=m.ROOT,stdout=log,stderr=subprocess.STDOUT)
            preview_check=readiness(18770,True)
            save('PREVIEW_ACCEPTANCE.json',dict(pid=preview.pid,checks=preview_check))
            preview.terminate();preview.wait(timeout=10)
        assert identity(old['pid'])==old and int(pane('#{pane_pid}'))==old['pid']
        for name,digest in sources.items():assert m.sha(HERE/name)==digest
        call('tmux','set-option','-w','-t',PANE,'remain-on-exit','on')
        assert identity(old['pid'])==old
        os.kill(old['pid'],signal.SIGTERM)  # Only the exactly verified old HTTP monitor.
        switched=True;wait_dead()
        call('tmux','respawn-pane','-t',PANE,'-c',str(m.ROOT),shlex.join(NEW_ARGV))
        new_pid=int(pane('#{pane_pid}'))
        checks=readiness(18769,True)
        new=identity(new_pid)
        assert new['argv']==NEW_ARGV and new['cwd']==old['cwd'] and new['uid']==old['uid']
        assert pane('#{pane_id}')==PANE and int(pane('#{pane_pid}'))==new_pid
        for p in protected:assert identity(p['pid'])==p,'PROTECTED_PRODUCER_CHANGED'
        for name,digest in inputs.items():assert m.sha(m.ROOT/name)==digest,'FROZEN_INPUT_CHANGED'
        call('tmux','set-option','-w','-t',PANE,'remain-on-exit',remain)
        save('DEPLOY_SUCCESS.json',dict(status='MONITOR_ONLY_SWITCH_VERIFIED',new_monitor=new,checks=checks,
            protected_producers_unchanged=True,frozen_inputs_unchanged=True,time_unix=time.time(),
            production_processes_signalled=0,old_monitor_pid_signalled=old['pid']))
        print(json.dumps(dict(status='MONITOR_ONLY_SWITCH_VERIFIED',pid=new_pid,port=18769,checks=checks)),flush=True)
    except BaseException as exc:
        receipt=dict(error=repr(exc),time_unix=time.time(),switched=switched)
        if switched:
            try:
                if pane('#{pane_dead}')=='0':
                    live=identity(int(pane('#{pane_pid}')))
                    assert live['argv']==NEW_ARGV and live['cwd']==str(m.ROOT),'UNKNOWN_MONITOR_PANE_NO_SIGNAL'
                    assert identity(live['pid'])==live
                    os.kill(live['pid'],signal.SIGTERM)
                wait_dead()
                call('tmux','respawn-pane','-t',PANE,'-c',str(m.ROOT),shlex.join(OLD_ARGV))
                receipt['rollback']=readiness(18769,False)
                call('tmux','set-option','-w','-t',PANE,'remain-on-exit',remain)
            except Exception as recovery:receipt['manual_recovery_required']=repr(recovery)
        save('DEPLOY_FAILURE.json',receipt)
        raise
    finally:
        if preview and preview.poll() is None:
            preview.terminate();preview.wait(timeout=10)

if __name__=='__main__':main()
