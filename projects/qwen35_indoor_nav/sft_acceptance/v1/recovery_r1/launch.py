"""Guarded GPU4 lease, bounded worker, and unconditional verified restoration."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent;LINE=OUT.parents[2];ROOT=LINE.parents[1]
TARGET='vla_idle_occupancy_20260904:1.0'
GPU=4;UUID='GPU-e458147b-4739-d22a-e764-50743cff4a11'
EXPECTED='.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu 4 --reserve-mib 2048 --max-initial-used-mib 1024 --tag vla_idle_occupancy_20260904'

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def call(*args):return subprocess.check_output(args,text=True,timeout=30).strip()
def pane(fmt):return call('tmux','display-message','-p','-t',TARGET,fmt)
def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
def gpu():
    x=ET.fromstring(call('nvidia-smi','-i',str(GPU),'-q','-x')).find('gpu');assert x.findtext('uuid')==UUID
    return dict(uuid=UUID,memory_mib=float(x.findtext('fb_memory_usage/used').split()[0]),processes=[{c.tag:c.text for c in p} for p in x.findall('processes/process_info')])
def env_for_run():
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','CONDA_PREFIX','LD_LIBRARY_PATH'):env.pop(k,None)
    env.update(PATH=str(LINE/'.envs/q35n_qwen_g2_v1/bin')+':/usr/bin:/bin',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(GPU),SFT_PHYSICAL_GPU=str(GPU),
        HF_HOME=str(OUT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        XDG_CACHE_HOME=str(OUT/'cache'),TORCH_HOME=str(OUT/'cache/torch'),TRITON_CACHE_DIR=str(OUT/'cache/triton'),CUDA_CACHE_PATH=str(OUT/'cache/cuda'),
        NUMBA_CACHE_DIR=str(OUT/'cache/numba'),MPLCONFIGDIR=str(OUT/'cache/matplotlib'),TMPDIR=str(OUT/'tmp'),TMP=str(OUT/'tmp'),TEMP=str(OUT/'tmp'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    return env
def stop_own(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def main():
    assert not (OUT/'LEASE_BEFORE.json').exists(),'No automatic lease/retry'
    for d in ['cache','cache/hf','cache/torch','cache/triton','cache/cuda','cache/numba','cache/matplotlib','tmp','checkpoints']:(OUT/d).mkdir(parents=True,exist_ok=True)
    # Record full code difference and all implementation files before any GPU work.
    original=LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/probe.py'
    with (OUT/'CODE_DIFF.patch').open('x') as f:f.writelines(difflib.unified_diff(original.read_text().splitlines(True),(OUT/'policy.py').read_text().splitlines(True),fromfile=str(original.relative_to(ROOT)),tofile='sft_acceptance/v1/policy.py'))
    save('CODE_LOCK.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
    save('CODE_PROVENANCE.json',dict(original_probe_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),differences=['Removed synthetic reader/query and BCE entirely','Retained G2 LoRA, tokenizer IDs, vision no_grad, native MRoPE and recurrent writer/action readout','Added strict white-list entry point, validated image decoding, bounded tokens and finite outputs','New action-only AdamW/TBPTT training, offline evaluation, trainable-state serialization and separate simulator service','No sealed code imported or executed; old GPU3 lease not used'],cpu_preparation_failures=[dict(command='/usr/bin/python3 -B prepare.py',reason='system Python3.6 does not support capture_output; no files written and no GPU used',resolution='used authorized project Python3.10, no package changes')]))
    before=gpu();pid=int(pane('#{pane_pid}'))
    assert pane('#{pane_current_path}')==str(ROOT) and command(pid)==EXPECTED and Path(f'/proc/{pid}/cwd').resolve()==ROOT
    others=[p for p in before['processes'] if int(p['pid'])!=pid]
    assert any(int(p['pid'])==pid for p in before['processes'])
    assert sum(float(p['used_memory'].split()[0]) for p in others)<1024
    for p in others:
        p['full_command']=command(int(p['pid']))
        assert '--device cuda:0' in p['full_command'] and 'eval.scripts.evaluate_pointgoal' in p['full_command']
    remain=call('tmux','show-options','-w','-v','-t',TARGET,'remain-on-exit')
    save('LEASE_BEFORE.json',dict(pid=pid,command=EXPECTED,cwd=str(ROOT),pane=TARGET,pane_id=pane('#{pane_id}'),remain_on_exit=remain,physical_gpu=GPU,gpu_uuid=UUID,other_contexts=others,
        restore_method='respawn exact verified command on same pane after own workers exit; cache environment redirected inside v1'))
    proc=None;leased=False;started=None;error=None;active_pid=None
    def interrupted(sig,frame):raise RuntimeError(f'Launcher interrupted: signal {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        subprocess.run(['tmux','set-option','-w','-t',TARGET,'remain-on-exit','on'],check=True)
        os.kill(pid,signal.SIGTERM);leased=True
        for _ in range(80):
            if not Path(f'/proc/{pid}').exists():break
            time.sleep(.25)
        assert not Path(f'/proc/{pid}').exists(),'Occupancy did not exit; no signal escalation'
        subprocess.run(['tmux','respawn-pane','-t',TARGET,'-c',str(ROOT),'sleep 24000'],check=True)
        active_pid=int(pane('#{pane_pid}'));save('LEASE_ACTIVE.json',dict(pid=active_pid,command='sleep 24000',pane=TARGET))
        time.sleep(2);save('GPU_BEFORE_WORKER.json',gpu())
        assert gpu()['memory_mib']<1024
        started=time.time();env=env_for_run()
        with (OUT/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(OUT/'run_sft.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('WORKER_PROCESS.json',dict(pid=proc.pid,started_unix=started,physical_gpu=GPU,logical_gpu=0))
            existing={int(p['pid']) for p in others}
            with (OUT/'RESOURCE_SAMPLES.jsonl').open('x') as f:
                while proc.poll() is None:
                    s=gpu();elapsed=time.time()-started
                    children=[]
                    cp=OUT/'CHILD_PROCESSES.jsonl'
                    if cp.exists():children=[int(json.loads(line)['pid']) for line in cp.read_text().splitlines()]
                    own={proc.pid,*children};rss=0
                    for own_pid in own:
                        path=Path(f'/proc/{own_pid}/status')
                        if path.exists():
                            for line in path.read_text().splitlines():
                                if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                    disk=sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file())
                    s.update(elapsed_seconds=elapsed,worker_tree_rss_bytes=rss,new_output_bytes=disk)
                    f.write(json.dumps(s)+'\n');f.flush()
                    assert s['memory_mib']<=28*1024,'GPU_MEMORY_CAP'
                    assert {int(p['pid']) for p in s['processes']}<=existing|own,'New unrelated GPU process: stop own worker'
                    assert rss<=48*1024**3 and disk<=20*1024**3 and elapsed<=20961.212619543076,'RESOURCE_CAP'
                    time.sleep(5)
            proc.wait()
            if proc.returncode:raise RuntimeError(f'Worker exited {proc.returncode}; no automatic retry')
    except BaseException as ex:
        error=repr(ex);save('LAUNCH_FAILURE.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        if proc:stop_own(proc)
        save('EXECUTION_RESULT.json',dict(returncode=proc.returncode if proc else None,error=error,gpu_stage_wall_seconds=time.time()-started if started else 0,real_tasks_stopped=0))
        save('GPU_AFTER_WORKER.json',gpu())
        if leased:
            if active_pid is not None:
                assert int(pane('#{pane_pid}'))==active_pid and command(active_pid)=='sleep 24000'
            assert gpu()['memory_mib']<1024,'Cannot restore on top of an active job'
            restore_env=env_for_run()
            # Do not propagate CUDA_VISIBLE_DEVICES: original occupant explicitly uses physical cuda:4.
            assignments=[f'{k}={restore_env[k]}' for k in ['PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR']]
            restore_cmd='env '+' '.join(assignments)+' '+EXPECTED
            subprocess.run(['tmux','respawn-pane','-k','-t',TARGET,'-c',str(ROOT),restore_cmd],check=True)
            subprocess.run(['tmux','set-option','-w','-t',TARGET,'remain-on-exit',remain],check=True)
            newpid=int(pane('#{pane_pid}'))
            for _ in range(60):
                time.sleep(.5)
                if Path(f'/proc/{newpid}').exists() and command(newpid)==EXPECTED and gpu()['memory_mib']>20000:break
            restored=Path(f'/proc/{newpid}').exists() and command(newpid)==EXPECTED and gpu()['memory_mib']>20000
            save('LEASE_RESTORED.json',dict(restored=restored,pid=newpid,command=EXPECTED,actual_restore_shell_command=restore_cmd,cwd=str(ROOT),pane=TARGET,physical_gpu=GPU,real_tasks_stopped=0,gpu_state=gpu()))
            assert restored,'Occupancy restoration failed'
    if error:raise RuntimeError(error)

if __name__=='__main__':main()

