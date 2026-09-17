"""Launch exactly one prepared correction run and its bounded review."""
import json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
T=LINE/'sft_acceptance/ordinary_policy_preservation_v9'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def ident(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,start=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),
      cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1])
def main():
    assert not (HERE/'LAUNCH_RECEIPT.json').exists() and not (T/'lease_v1').exists() and not (T/'formal').exists()
    assert json.loads((HERE/'CPU_PREFLIGHT.json').read_text())['status']=='PASS'
    assert json.loads((T/'PROTOCOL_FILESTORE.json').read_text())['accounting']['deadline_unix']-time.time()>2200
    owners=[]
    for gpu,pid,start,pane in [(3,3031959,163924910,'%153'),(4,3032156,163925372,'%149'),(5,3032241,163925704,'%235')]:
        v=ident(pid);assert v['start']==start and v['cwd']==str(ROOT)
        assert v['argv']==['.envs/etpr1/bin/python','-u','scripts/occupy_idle_gpu.py','--gpu',str(gpu),'--reserve-mib','2048','--max-initial-used-mib','1536' if gpu==3 else '1024','--tag','vla_idle_occupancy_20260904']
        assert int(subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{pane_pid}'],text=True))==pid
        owners.append(v)
    cmd=f'exec {PY} -I -S -B {T}/lease_run.py {T}/RUNBOOK.json'
    subprocess.run(['tmux','new-session','-d','-s','q35n_policy_preservation_v9','-c',str(ROOT),cmd],check=True)
    m=runpy.run_path(str(HERE/'workflow.py'));m['freeze']()
    subprocess.run(['tmux','new-session','-d','-s','q35n_policy_preservation_v9_review','-c',str(ROOT),
       f'exec {PY} -I -S -B {HERE}/workflow.py run'],check=True)
    with (HERE/'LAUNCH_RECEIPT.json').open('x') as f:json.dump(dict(unix=time.time(),holders_before=owners,one_training=True,one_fixed_final_eval=True,foreign_processes_signaled=[]),f,indent=2)
    print('ONE_TRAIN_AND_FINAL_REVIEW_LAUNCHED')
if __name__=='__main__':main()
