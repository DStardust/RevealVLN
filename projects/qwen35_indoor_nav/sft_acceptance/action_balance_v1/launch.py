"""Lease only verified GPU5/6 holders; always clean ranks and restore holders."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
GPUS=[5,6]
PANES={5:'vla_idle_occupancy_20260904:2.0',6:'vla_idle_occupancy_20260904:3.0'}
UUIDS={5:'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b',6:'GPU-a7120b0e-d348-23d4-ffe4-3072745b8490'}

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def call(*args):return subprocess.check_output(args,text=True,timeout=30).strip()
def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
def expected(g):return f'.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu {g} --reserve-mib 2048 --max-initial-used-mib 1024 --tag vla_idle_occupancy_20260904'
def pane(g,fmt):return call('tmux','display-message','-p','-t',PANES[g],fmt)

def gpu(g):
    x=ET.fromstring(call('nvidia-smi','-i',str(g),'-q','-x')).find('gpu')
    assert x.findtext('uuid')==UUIDS[g]
    return dict(gpu=g,uuid=UUIDS[g],memory_mib=float(x.findtext('fb_memory_usage/used').split()[0]),
                processes=[{c.tag:c.text for c in p} for p in x.findall('processes/process_info')])

def env_for(rank):
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','CONDA_PREFIX','LD_LIBRARY_PATH'):env.pop(k,None)
    cache=OUT/f'cache/rank{rank}';tmp=OUT/f'tmp/rank{rank}'
    cache.mkdir(parents=True,exist_ok=True);tmp.mkdir(parents=True,exist_ok=True)
    env.update(PATH=str(LINE/'.envs/q35n_qwen_g2_v1/bin')+':/usr/bin:/bin',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=str(GPUS[rank]),RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK='0',
        HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    return env

def stop_own(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def main():
    assert not (OUT/'LEASE_BEFORE.json').exists(),'No automatic rerun'
    lock=json.loads((OUT/'LOCK.json').read_text())
    for rel,h in lock['sources'].items():assert hashlib.sha256((LINE/rel).read_bytes()).hexdigest()==h,rel
    for name,h in lock['code'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'SUBSET.json').read_bytes()).hexdigest()==lock['subset_sha256']
    assert hashlib.sha256((OUT/'WEIGHTS.json').read_bytes()).hexdigest()==lock['weights_sha256']
    before={}
    for g in GPUS:
        s=gpu(g);pid=int(pane(g,'#{pane_pid}'))
        assert command(pid)==expected(g) and Path(f'/proc/{pid}/cwd').resolve()==ROOT
        assert pane(g,'#{pane_current_path}')==str(ROOT)
        others=[p for p in s['processes'] if int(p['pid'])!=pid]
        assert any(int(p['pid'])==pid for p in s['processes'])
        assert sum(float(p['used_memory'].split()[0]) for p in others)<1024
        # Preserve these small pre-existing foreign CUDA contexts; never signal them.
        for p in others:
            p['command']=command(int(p['pid']))
            assert 'eval.scripts.evaluate_pointgoal' in p['command'] and '--device cuda:0' in p['command']
        before[g]=dict(pid=pid,command=expected(g),cwd=str(ROOT),pane=PANES[g],pane_id=pane(g,'#{pane_id}'),
            remain=call('tmux','show-options','-w','-v','-t',PANES[g],'remain-on-exit'),gpu=s,other_contexts=others)
    save('LEASE_BEFORE.json',before)
    leased=[];active={};procs=[];logs=[];error=None;start=time.time();restoration={}
    def interrupt(sig,frame):raise RuntimeError(f'Interrupted signal {sig}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for g in GPUS:
            subprocess.run(['tmux','set-option','-w','-t',PANES[g],'remain-on-exit','on'],check=True)
            pid=before[g]['pid']
            assert command(pid)==expected(g)
            os.kill(pid,signal.SIGTERM);leased.append(g)
            for _ in range(80):
                if not Path(f'/proc/{pid}').exists():break
                time.sleep(.25)
            assert not Path(f'/proc/{pid}').exists(),'Holder did not exit; no kill escalation'
            subprocess.run(['tmux','respawn-pane','-t',PANES[g],'-c',str(ROOT),'sleep 24000'],check=True)
            active[g]=int(pane(g,'#{pane_pid}'))
            assert gpu(g)['memory_mib']<1024
        save('LEASE_ACTIVE.json',active)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        for rank in range(2):
            env=env_for(rank);env.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
            log=(OUT/f'rank{rank}.log').open('x');logs.append(log)
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(OUT/'worker.py')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append(proc)
        save('WORKERS.json',dict(pids=[p.pid for p in procs],GPUS=GPUS,started_unix=start,master_port=port))
        with (OUT/'RESOURCES.jsonl').open('x') as f:
            while any(p.poll() is None for p in procs):
                for p in procs:
                    assert p.poll() in (None,0), f'Rank failure {p.pid}: {p.returncode}'
                ss=[gpu(g) for g in GPUS];rss=0
                for p in procs:
                    path=Path(f'/proc/{p.pid}/status')
                    if path.exists():
                        for line in path.read_text().splitlines():
                            if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                elapsed=time.time()-start
                disk=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
                f.write(json.dumps(dict(unix=time.time(),wall=elapsed,rss=rss,disk=disk,gpus=ss))+'\n');f.flush()
                for rank,s in enumerate(ss):
                    other={int(p['pid']) for p in before[GPUS[rank]]['other_contexts']}
                    assert {int(p['pid']) for p in s['processes']}<=other|{procs[rank].pid},'New unrelated GPU task'
                    assert s['memory_mib']<=28*1024,'GPU memory cap'
                assert elapsed<=7200 and rss<=48*1024**3 and disk<=10*1024**3,'Resource cap'
                time.sleep(5)
        assert all(p.returncode==0 for p in procs)
    except BaseException as ex:
        error=repr(ex);save('LAUNCH_FAILURE.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        for p in procs:stop_own(p)
        for log in logs:log.close()
        # Attempt each restoration independently; failure of one must not skip the other.
        for g in leased:
            try:
                assert gpu(g)['memory_mib']<1024,'Cannot restore over an active/new job'
                if g in active:
                    assert int(pane(g,'#{pane_pid}'))==active[g] and command(active[g])=='sleep 24000'
                else:
                    assert pane(g,'#{pane_dead}')=='1'
                e=env_for(GPUS.index(g))
                assigns=[f'{k}={e[k]}' for k in ['PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR']]
                restore='env '+' '.join(shlex.quote(s) for s in assigns)+' '+expected(g)
                subprocess.run(['tmux','respawn-pane','-k','-t',PANES[g],'-c',str(ROOT),restore],check=True)
                subprocess.run(['tmux','set-option','-w','-t',PANES[g],'remain-on-exit',before[g]['remain']],check=True)
                pid=int(pane(g,'#{pane_pid}'))
                for _ in range(60):
                    time.sleep(.5)
                    if Path(f'/proc/{pid}').exists() and command(pid)==expected(g) and gpu(g)['memory_mib']>20000:break
                passed=Path(f'/proc/{pid}').exists() and command(pid)==expected(g) and gpu(g)['memory_mib']>20000
                restoration[g]=dict(restored=passed,pid=pid,command=expected(g),pane=PANES[g],gpu=gpu(g))
            except BaseException as ex:restoration[g]=dict(restored=False,error=repr(ex),traceback=traceback.format_exc())
        save('RESTORATION.json',restoration)
        save('EXECUTION.json',dict(error=error,returncodes=[p.returncode for p in procs],wall_seconds=time.time()-start,
            GPU_count=len(leased),conservative_GPU_seconds=len(leased)*(time.time()-start),real_tasks_stopped=0,
            all_leased_holders_restored=all(restoration[g]['restored'] for g in leased)))
    if error:raise RuntimeError(error)
    assert all(restoration[g]['restored'] for g in leased)

if __name__=='__main__':main()
