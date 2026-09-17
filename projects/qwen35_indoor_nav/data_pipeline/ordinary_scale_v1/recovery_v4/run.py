"""Lease exact GPU5 holder only; all-process XML watchdog; restore in finally."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
LINE=BASE.parents[1]
ROOT=BASE.parents[3]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'
PANE='vla_idle_occupancy_20260904:2.0'
PANE_ID='%150'
HOLDER_PID=2990950
UUID='GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b'
EXPECTED='.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu 5 --reserve-mib 2048 --max-initial-used-mib 1024 --tag vla_idle_occupancy_20260904'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

size=load('v4_size',HERE/'safe_size.py')
preflight=load('v4_preflight',HERE/'preflight.py')

def call(*args):return subprocess.check_output(args,text=True,timeout=30).strip()
def pane(fmt):return call('tmux','display-message','-p','-t',PANE,fmt)
def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
def save(path,obj):
    with path.open('x') as handle:json.dump(obj,handle,indent=2,allow_nan=False)

def parse_gpu(xml):
    node=ET.fromstring(xml).find('gpu')
    assert node.findtext('uuid')==UUID,'GPU_UUID'
    rows=[{child.tag:child.text for child in p} for p in node.findall('processes/process_info')]
    processes={int(r['pid']):float(r['used_memory'].split()[0]) for r in rows}
    return dict(gpu=5,uuid=UUID,memory_mib=float(node.findtext('fb_memory_usage/used').split()[0]),
                utilization=float(node.findtext('utilization/gpu_util').split()[0]),processes=processes,process_rows=rows)

def gpu():return parse_gpu(call('nvidia-smi','-i','5','-q','-x'))

def validate_contexts(snapshot,own_pid=None):
    external={pid:mib for pid,mib in snapshot['processes'].items() if pid!=own_pid}
    assert all(0<=mib<=768 for mib in external.values()),'EXTERNAL_CONTEXT_PER_PID_CAP'
    assert sum(external.values())<=2048,'EXTERNAL_CONTEXT_TOTAL_CAP'
    if own_pid is not None:
        upper=snapshot['memory_mib']-sum(external.values())
        assert snapshot['processes'].get(own_pid,0)<4096 and upper<4096,'OWN_GPU_UPPER_BOUND_CAP'
    else:assert snapshot['memory_mib']<1024,'RESTORABLE_IDLE_GPU_CAP'
    return external

def stop_own(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def main():
    producer=(BASE/'PRODUCER.lock').open('a')
    fcntl.flock(producer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=preflight.verify()
    approval=json.loads((HERE/'MAIN_AGENT_APPROVAL.json').read_text())
    assert approval==dict(approved=True,gpu=5,input_lock_sha256=hashlib.sha256((HERE/'INPUT_LOCK.json').read_bytes()).hexdigest())
    out=HERE/'run';out.mkdir(exist_ok=False)
    before=gpu()
    assert int(pane('#{pane_pid}'))==HOLDER_PID and pane('#{pane_id}')==PANE_ID
    assert command(HOLDER_PID)==EXPECTED and Path(f'/proc/{HOLDER_PID}/cwd').resolve()==ROOT
    assert pane('#{pane_current_path}')==str(ROOT)
    assert HOLDER_PID in before['processes']
    others={pid:mib for pid,mib in before['processes'].items() if pid!=HOLDER_PID}
    assert all(mib<=768 for mib in others.values()) and sum(others.values())<1024
    remain=call('tmux','show-options','-w','-v','-t',PANE,'remain-on-exit')
    save(out/'LEASE_BEFORE.json',dict(gpu=before,pid=HOLDER_PID,pane=PANE,pane_id=PANE_ID,
        command=EXPECTED,cwd=str(ROOT),remain=remain,external_contexts=others))
    cache=out/'cache';cache.mkdir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),NUMBA_CACHE_DIR=str(cache/'numba'),
        MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    started=time.monotonic();proc=None;leased=False;sleeper=None;error=None;restoration={};samples=0
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        assert size.measure(BASE)['apparent_bytes_conservative']<199*1024**3
        call('tmux','set-option','-w','-t',PANE,'remain-on-exit','on')
        assert command(HOLDER_PID)==EXPECTED and int(pane('#{pane_pid}'))==HOLDER_PID
        os.kill(HOLDER_PID,signal.SIGTERM);leased=True
        for _ in range(80):
            if not Path(f'/proc/{HOLDER_PID}').exists():break
            time.sleep(.25)
        assert not Path(f'/proc/{HOLDER_PID}').exists(),'HOLDER_EXIT_FAILED_NO_ESCALATION'
        call('tmux','respawn-pane','-t',PANE,'-c',str(ROOT),'sleep 24000')
        sleeper=int(pane('#{pane_pid}'))
        idle=gpu()
        for _ in range(20):
            validate_contexts(idle)
            if idle['utilization']==0:break
            time.sleep(.5);idle=gpu()
        save(out/'LEASE_ACTIVE.json',dict(sleeper=sleeper,gpu=idle))
        validate_contexts(idle)
        assert idle['utilization']==0,'GPU_NOT_IDLE_AFTER_LEASE'
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(HERE/'worker.py')],
                cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(out/'PROCESS.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=time.time(),gpu=5))
            while proc.poll() is None:
                time.sleep(5);samples+=1
                snapshot=gpu();elapsed=time.monotonic()-started
                with (out/'GPU_SNAPSHOTS.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(snapshot,elapsed=elapsed))+'\n')
                validate_contexts(snapshot,proc.pid)
                assert elapsed+lock['previous_wall_seconds']<14340,'TOTAL_WALL_CAP_WITH_CLEANUP_MARGIN'
                rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                assert rss.returncode==0 or proc.poll() is not None,'RSS_MONITOR_ERROR'
                assert int(rss.stdout.strip() or 0)<12*1024**2,'RAM_CAP'
                if samples%12==0:
                    disk=size.measure(BASE)
                    with (out/'DISK_SNAPSHOTS.jsonl').open('a') as handle:handle.write(json.dumps(disk)+'\n')
                    assert disk['apparent_bytes_conservative']<199*1024**3,'DISK_CAP'
    except BaseException as ex:error=repr(ex)
    finally:
        # A second interactive signal must not skip holder restoration.
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:stop_own(proc)
        except BaseException as ex:error=(error or '')+';CLEANUP:'+repr(ex)
        if leased:
            try:
                snapshot=gpu();save(out/'GPU_PRE_RESTORE.json',snapshot)
                validate_contexts(snapshot)
                assert pane('#{pane_id}')==PANE_ID
                if sleeper is not None:
                    assert int(pane('#{pane_pid}'))==sleeper and command(sleeper)=='sleep 24000'
                else:assert pane('#{pane_dead}')=='1'
                call('tmux','respawn-pane','-k','-t',PANE,'-c',str(ROOT),EXPECTED)
                call('tmux','set-option','-w','-t',PANE,'remain-on-exit',remain)
                restored_pid=int(pane('#{pane_pid}'))
                passed=False
                for _ in range(60):
                    time.sleep(.5)
                    if not Path(f'/proc/{restored_pid}').exists():break
                    snapshot=gpu()
                    if command(restored_pid)==EXPECTED and snapshot['processes'].get(restored_pid,0)>20000:
                        passed=True;break
                restoration=dict(restored=passed,pid=restored_pid,command=EXPECTED,gpu=snapshot)
            except BaseException as ex:restoration=dict(restored=False,error=repr(ex),external_tasks_stopped=0)
        else:restoration=dict(restored=True,not_borrowed=True)
        save(out/'RESTORATION.json',restoration)
        result=dict(error=error,returncode=proc.returncode if proc else None,wall_seconds=time.monotonic()-started,
            previous_wall_seconds=lock['previous_wall_seconds'],holders_touched=leased,
            holder_restoration_required=leased,holder_restored=restoration.get('restored',False),
            external_processes_stopped=0,scientific_pass=False)
        save(out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if error or not proc or proc.returncode or not restoration.get('restored'):raise SystemExit(1)

if __name__=='__main__':main()
