"""Exactly verified holder lease per lane; two sequential isolated shard workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import safe_size


def call(*args):return subprocess.check_output(args,text=True,timeout=30).strip()


def process_identity(pid):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime_ticks=int(fields[19]),proc_uid=proc.stat().st_uid,
        cwd=str((proc/'cwd').resolve()),cmdline=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))


def pane(identity,fmt):return call('tmux','display-message','-p','-t',identity['pane_target'],fmt)


def verify_identity(identity):
    actual=process_identity(identity['pid'])
    for key in actual:assert actual[key]==identity[key],('HOLDER_IDENTITY',key)
    assert pane(identity,'#{pane_id}')==identity['pane_id']
    assert int(pane(identity,'#{pane_pid}'))==identity['pid']
    assert pane(identity,'#{pane_dead}')=='0'
    assert pane(identity,'#{pane_current_path}')==str(c.ROOT)
    return actual


def same_process_running(pid,starttime_ticks):
    """Exit wait uses stat only: cwd/cmdline may disappear before /proc does."""
    try:fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    assert int(fields[19])==starttime_ticks,'PID_REUSED_DURING_EXIT'
    return fields[0] not in ('Z','X','x')


def wait_holder_exit(identity):
    for _ in range(80):
        if not same_process_running(identity['pid'],identity['starttime_ticks']):return
        time.sleep(.25)
    raise AssertionError('HOLDER_EXIT_NO_ESCALATION')


def wait_pane_dead(identity):
    for _ in range(40):
        assert pane(identity,'#{pane_id}')==identity['pane_id'],'PANE_ID_CHANGED'
        if pane(identity,'#{pane_dead}')=='1':return
        time.sleep(.25)
    raise AssertionError('PANE_DEAD_TRANSITION_TIMEOUT')


def drain_context(identity,released_pid,out,label):
    """Allow only the already-released owned PID to drain; save every raw sample."""
    for _ in range(80):
        snapshot=gpu_snapshot(identity)
        with (out/f'{label}_DRAIN.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
        external={p:m for p,m in snapshot['processes'].items() if p!=released_pid}
        assert all(0<=m<=768 for m in external.values()) and sum(external.values())<=2048,'NEW_EXTERNAL_DURING_DRAIN'
        if released_pid not in snapshot['processes'] and snapshot['memory_mib']<1024:
            contexts(snapshot,restorable=True);return snapshot
        time.sleep(.25)
    raise AssertionError('RELEASED_CONTEXT_DRAIN_TIMEOUT')


def restore_holder(identity,sleeper_identity,remain,out,released_pid=None):
    after=drain_context(identity,released_pid if released_pid is not None else identity['pid'],out,'PRE_RESTORE')
    c.save(out/'GPU_PRE_RESTORE.json',after)
    assert pane(identity,'#{pane_id}')==identity['pane_id']
    if sleeper_identity is not None:
        pid=sleeper_identity['pid'];assert int(pane(identity,'#{pane_pid}'))==pid
        if same_process_running(pid,sleeper_identity['starttime_ticks']):
            try:
                assert process_identity(pid)==sleeper_identity,'SLEEPER_IDENTITY_CHANGED'
                assert sleeper_identity['cmdline']==['sleep','24000']
            except FileNotFoundError:
                if same_process_running(pid,sleeper_identity['starttime_ticks']):raise
            else:
                try:os.kill(pid,signal.SIGTERM)  # exactly verified own temporary sleeper only
                except ProcessLookupError:pass
        wait_holder_exit(sleeper_identity)
        wait_pane_dead(identity)
    else:wait_pane_dead(identity)
    call('tmux','respawn-pane','-t',identity['pane_target'],'-c',identity['cwd'],shlex.join(identity['cmdline']))
    pid=int(pane(identity,'#{pane_pid}'));passed=False;new_identity=None;snapshot=after
    for _ in range(60):
        time.sleep(.5)
        try:new_identity=process_identity(pid)
        except FileNotFoundError:continue  # startup/exit race is not permission to delete pane
        snapshot=gpu_snapshot(identity)
        if (new_identity['cmdline']==identity['cmdline'] and new_identity['cwd']==identity['cwd']
                and new_identity['proc_uid']==identity['proc_uid'] and snapshot['processes'].get(pid,0)>20000):
            passed=True;break
    # Keep remain-on-exit ON if startup fails. Never delete the only recovery pane.
    if passed:call('tmux','set-option','-w','-t',identity['pane_target'],'remain-on-exit',remain)
    return dict(restored=passed,new_identity=new_identity,gpu=snapshot,recovery_pane_preserved_if_failed=not passed)


def parse_gpu(xml,gpu,uuid):
    nodes=ET.fromstring(xml).findall('gpu');assert len(nodes)==1
    node=nodes[0];assert node.findtext('uuid')==uuid,'GPU_UUID'
    rows=[{x.tag:x.text for x in p} for p in node.findall('./processes/process_info')]
    processes={int(r['pid']):float(r['used_memory'].split()[0]) for r in rows}
    assert len(processes)==len(rows),'DUPLICATE_GPU_PID'
    return dict(gpu=gpu,uuid=uuid,memory_mib=float(node.findtext('fb_memory_usage/used').split()[0]),
        utilization=float(node.findtext('utilization/gpu_util').split()[0]),processes=processes,process_rows=rows)


def gpu_snapshot(identity):
    return parse_gpu(call('nvidia-smi','-i',str(identity['gpu_device']),'-q','-x'),identity['gpu_device'],identity['gpu_uuid'])


def contexts(snapshot,own=None,restorable=False):
    external={pid:mib for pid,mib in snapshot['processes'].items() if pid!=own}
    assert all(0<=m<=768 for m in external.values()) and sum(external.values())<=2048,'EXTERNAL_GPU_RESOURCE'
    assert sum(snapshot['processes'].values())<=snapshot['memory_mib'],'GPU_MEMORY_ACCOUNTING'
    upper=snapshot['memory_mib']-sum(external.values())
    if own is not None:assert 0<=upper<4096,'OWN_GPU_UPPER_BOUND'
    if restorable:assert snapshot['memory_mib']<1024,'RESTORE_NOT_SAFE_EXTERNAL_LOAD'
    return upper


def stop_own(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def disk_guard(shard):
    value=safe_size.measure(c.shard_root(shard))
    # 1GiB early-stop margin within the independently allocated 49GiB cap.
    assert value['apparent_bytes_conservative']<48*1024**3,'SHARD_DISK_EARLY_STOP'
    return value


def execute(gpu):
    started=time.monotonic();c.approved(gpu)
    lane=HERE/'lanes'/f'gpu_{gpu}';lane.mkdir(parents=True,exist_ok=True)
    lane_lock=(lane/'PRODUCER.lock').open('a');fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=[]
    for p in sorted(lane.glob('attempt_*')):
        assert (p/'RESULT.json').exists(),'UNCLOSED_LANE_ATTEMPT_REQUIRES_RECONCILIATION'
        old.append(c.read(p/'RESULT.json'))
    previous_seconds=sum(r['wall_seconds'] for r in old)
    spent={s:sum(w['wall_seconds'] for r in old for w in r['workers'] if w['shard']==s) for s in c.LANES[gpu]}
    assert previous_seconds<7080,'LANE_BUDGET_EXHAUSTED'
    pending=[]
    for shard in c.LANES[gpu]:
        if not c.state(shard)['complete']:pending.append(shard)
    assert pending,'ALL_LANE_SHARDS_CLOSED_NO_RERUN'
    out=lane/f'attempt_{len(old):03d}';out.mkdir(exist_ok=False)
    identity=next(r for r in c.read(c.IDENTITIES)['holders'] if r['gpu_device']==gpu)
    if old:
        restore=old[-1]['restoration']
        assert restore.get('restored') and restore.get('new_identity'),'UNVERIFIED_PREVIOUS_RESTORATION'
        identity=dict(identity,**restore['new_identity'])
    c.save(out/'EXECUTION_CONFIG.json',dict(c.read(HERE/'PREPARED_CONFIG.json'),
        runtime_allowed=True,executable=True,gpu=gpu,pending_shards=pending,previous_lane_seconds=previous_seconds,
        previous_worker_seconds=spent,holder_identity=identity,approval_sha256=c.sha(HERE/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')))
    cache=out/'cache';cache.mkdir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
    env.update(PATH=f'{c.ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),NUMBA_CACHE_DIR=str(cache/'numba'),
        MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    leased=False;sleeper=None;sleeper_identity=None;proc=None;error=None;workers=[];remain=None;restoration={}
    def interrupt(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        before=gpu_snapshot(identity);verify_identity(identity)
        assert identity['pid'] in before['processes'],'HOLDER_GPU_MEMBERSHIP'
        external={p:m for p,m in before['processes'].items() if p!=identity['pid']}
        assert all(0<=m<=768 for m in external.values()) and sum(external.values())<1024,'GPU_NOT_EXCLUSIVELY_BORROWABLE'
        remain=call('tmux','show-options','-w','-v','-t',identity['pane_target'],'remain-on-exit')
        c.save(out/'LEASE_BEFORE.json',dict(identity=identity,gpu=before,remain=remain))
        for shard in pending:disk_guard(shard)
        call('tmux','set-option','-w','-t',identity['pane_target'],'remain-on-exit','on')
        verify_identity(identity)  # PID start/cwd/uid/argv/pane immediately before only authorized signal.
        leased=True
        try:os.kill(identity['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
        wait_holder_exit(identity)
        wait_pane_dead(identity)
        call('tmux','respawn-pane','-t',identity['pane_target'],'-c',str(c.ROOT),'sleep 24000')
        sleeper=int(pane(identity,'#{pane_pid}'));sleeper_identity=process_identity(sleeper)
        idle=drain_context(identity,identity['pid'],out,'HOLDER_RELEASE')
        for _ in range(20):
            contexts(idle,restorable=True)
            if idle['utilization']==0:break
            time.sleep(.5);idle=gpu_snapshot(identity)
        assert idle['utilization']==0,'GPU_NOT_IDLE_AFTER_LEASE'
        c.save(out/'LEASE_ACTIVE.json',dict(gpu=idle,sleeper=sleeper_identity))
        for shard in pending:
            assert time.monotonic()-started+previous_seconds<7080,'LANE_WALL_BUDGET'
            assert spent[shard]<3480,'SHARD_WALL_BUDGET'
            worker_started=time.monotonic();proc=None
            try:
                with (out/f'worker_{shard}.log').open('x') as log:
                    proc=subprocess.Popen([str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'worker.py'),
                        '--shard',str(shard),'--attempt',str(out)],cwd=c.LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    c.save(out/f'PROCESS_{shard}.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=time.time(),gpu=gpu,shard=shard))
                    samples=0
                    while proc.poll() is None:
                        time.sleep(5);samples+=1
                        snapshot=gpu_snapshot(identity)
                        with (out/'GPU_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(dict(snapshot,shard=shard,elapsed=time.monotonic()-started))+'\n')
                        contexts(snapshot,proc.pid)
                        assert time.monotonic()-started+previous_seconds<7080,'LANE_WALL_BUDGET'
                        assert time.monotonic()-worker_started+spent[shard]<3480,'SHARD_WALL_BUDGET'
                        rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],text=True,capture_output=True)
                        assert rss.returncode==0 or proc.poll() is not None,'RSS_READ'
                        assert int(rss.stdout.strip() or 0)<12*1024**2,'RAM_BUDGET'
                        if samples%6==0:
                            disk=disk_guard(shard)
                            assert safe_size.measure(HERE)['apparent_bytes_conservative']<3*1024**3,'METADATA_EARLY_STOP'
                            with (out/'DISK_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(dict(disk,shard=shard))+'\n')
                assert proc.returncode==0,'WORKER_FAILED'
                assert c.state(shard)['complete'],'WORKER_EXIT_WITHOUT_SHARD_CLOSURE'
                disk_guard(shard)
            finally:
                stop_own(proc)
                workers.append(dict(shard=shard,returncode=proc.returncode if proc else None,wall_seconds=time.monotonic()-worker_started))
                for name in ('PROGRESS.json','COUNTS_LAST_INVOCATION.json','GENERATION_COMPLETE.json'):
                    p=c.shard_root(shard)/name
                    if p.exists():c.save(out/f'SHARD_{shard}_{name}',c.read(p))
    except BaseException as exc:error=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:stop_own(proc)
        except BaseException as exc:error=(error or '')+'; OWN_CLEANUP:'+repr(exc)
        if leased:
            try:
                restoration=restore_holder(identity,sleeper_identity,remain,out,proc.pid if proc else None)
            except BaseException as exc:restoration=dict(restored=False,error=repr(exc),external_processes_stopped=0)
        else:
            restoration=dict(restored=True,not_borrowed=True)
            if remain is not None:
                try:
                    verify_identity(identity)
                    call('tmux','set-option','-w','-t',identity['pane_target'],'remain-on-exit',remain)
                except BaseException as exc:error=(error or '')+'; PANE_OPTION:'+repr(exc)
        c.save(out/'RESTORATION.json',restoration)
        result=dict(error=error,workers=workers,wall_seconds=time.monotonic()-started,
            previous_lane_seconds=previous_seconds,restoration=restoration,holders_touched=leased,
            external_processes_stopped=0,scientific_pass=False)
        c.save(out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if error or not restoration.get('restored'):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[6,7],required=True);execute(p.parse_args().gpu)
