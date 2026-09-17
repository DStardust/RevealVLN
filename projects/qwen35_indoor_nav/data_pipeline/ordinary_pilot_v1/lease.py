import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
TARGET='vla_idle_occupancy_20260904:0.0'
EXPECTED='.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu 3 --reserve-mib 2048 --max-initial-used-mib 1536 --tag vla_idle_occupancy_20260904'


def call(*args):return subprocess.check_output(args,text=True).strip()
def pane(fmt):return call('tmux','display-message','-p','-t',TARGET,fmt)
def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def acquire():
    assert pane('#{pane_current_path}')==str(ROOT)
    pid=int(pane('#{pane_pid}'));assert command(pid)==EXPECTED
    assert Path(f'/proc/{pid}/cwd').resolve()==ROOT
    assert (ROOT/'.envs/etpr1/bin/python').resolve().is_relative_to(ROOT)
    data=call('nvidia-smi','-i','3','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits')
    rows=[(int(x.split(',')[0]),int(x.split(',')[1])) for x in data.splitlines()]
    others=[(p,m) for p,m in rows if p!=pid]
    assert any(p==pid for p,m in rows) and all(m<=512 for p,m in others) and sum(m for p,m in others)<1024
    for p,m in others:
        args=command(p);assert '--device cuda:0' in args and 'eval.scripts.evaluate_pointgoal' in args
    remain=call('tmux','show-options','-w','-v','-t',TARGET,'remain-on-exit')
    save('LEASE_BEFORE.json',{'pid':pid,'target':TARGET,'pane_id':pane('#{pane_id}'),'command':EXPECTED,'cwd':str(ROOT),'other_contexts':others,'remain_on_exit':remain})
    # Graceful stop of this exact verified reservation process only.
    # respawn-pane installs a waiting lease after its normal SIGTERM handler exits.
    subprocess.run(['tmux','set-option','-w','-t',TARGET,'remain-on-exit','on'],check=True)
    os.kill(pid,15)
    for _ in range(80):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.25)
    assert not Path(f'/proc/{pid}').exists(),'Reservation did not stop; do not escalate signals'
    subprocess.run(['tmux','respawn-pane','-t',TARGET,'-c',str(ROOT),'sleep 7200'],check=True)
    save('LEASE_ACTIVE.json',{'pid':int(pane('#{pane_pid}')),'command':'sleep 7200'})
    time.sleep(2)


def restore():
    before=json.loads((OUT/'LEASE_BEFORE.json').read_text())
    active=json.loads((OUT/'LEASE_ACTIVE.json').read_text())
    assert pane('#{pane_id}')==before['pane_id'] and int(pane('#{pane_pid}'))==active['pid']
    assert command(active['pid'])=='sleep 7200'
    data=call('nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits')
    assert int(data)<1536,'Actual task present; must not restore on top of it'
    subprocess.run(['tmux','respawn-pane','-k','-t',TARGET,'-c',str(ROOT),EXPECTED],check=True)
    subprocess.run(['tmux','set-option','-w','-t',TARGET,'remain-on-exit',before['remain_on_exit']],check=True)
    pid=int(pane('#{pane_pid}'))
    for _ in range(40):
        time.sleep(.5)
        if Path(f'/proc/{pid}').exists() and command(pid)==EXPECTED:
            used=int(call('nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'))
            if used>20000:break
    assert command(pid)==EXPECTED and used>20000
    save('LEASE_RESTORED.json',{'restored':True,'pid':pid,'command':EXPECTED,'gpu_memory_mib':used,'real_tasks_stopped':0})
