"""Publish only this diagnosis and its evidence; never stage unrelated working files."""
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,BASE,HOLDOUT,V16,LINE,read,write,sha,load

def main():
    root=LINE.parents[1];cfg=read(HOLDOUT/'PROTOCOL.json');branch=cfg['publication_branch']
    helper=load('diagnosis_publication',BASE/'gpu_runtime_r1/publish_r2.py')
    paths=[p for p in HERE.glob('*') if p.is_file() and p.suffix in helper.TEXT and p.name not in ('PUSH_RECEIPT.json','EVIDENCE_MANIFEST.json')]
    paths.extend(p for p in (HERE/'runs/diagnosis_001').iterdir() if p.is_file())
    for job in ('monotonic-diagnosis-20260921-01','monotonic-fit-probe-20260921-01'):
        folder=HERE/'standalone_jobs'/job
        if read(folder/'STATUS.json')['status']!='COMPLETE':raise ValueError('DIAGNOSIS_NOT_COMPLETE')
        paths.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in helper.TEXT)
    for p in paths:helper.blob(p)
    write(HERE/'EVIDENCE_MANIFEST.json',dict(files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)},
        boundary='Derived CPU diagnosis only; old source/result/weights remain read-only. Raw licensed data stays local.',
        source_holdout_commit='360a14521749217cb87f0e0021f1984698b63e2d'))
    paths.append(HERE/'EVIDENCE_MANIFEST.json')
    with (V16/'b2_kl_repair_v1/PUBLISH.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!=branch:raise ValueError('BRANCH_CHANGED')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip():raise ValueError('OTHER_STAGED_CHANGES')
        subprocess.run(['git','add','-f','--',*[str(p.relative_to(root)) for p in paths]],cwd=root,check=True)
        subprocess.run(['git','commit','-m','Localize MONOTONIC seed instability with sealed rollout and frozen FIT CPU evidence'],cwd=root,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        proxy=shlex.join([cfg['standalone_python'],'-I','-S','-B',str(LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'),'%h','%p'])
        env=dict(os.environ,GIT_SSH_COMMAND=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=ssh.github.com','-p','443','-o','ProxyCommand='+proxy]))
        def remote(args):return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(args)],cwd=root,env=env,capture_output=True,text=True,timeout=1800)
        r=remote(['git','push','-u','origin',branch])
        if r.returncode:raise RuntimeError('PUSH_FAILED:'+r.stderr[-1200:])
        r=remote(['git','ls-remote','--heads','origin','refs/heads/'+branch])
        if r.returncode or not r.stdout.split() or r.stdout.split()[0]!=commit:raise ValueError('REMOTE_SHA_MISMATCH')
        receipt=dict(status='PUSHED_AND_VERIFIED',commit=commit,branch=branch,unix=time.time())
        write(HERE/'PUSH_RECEIPT.json',receipt);print(receipt,flush=True)

if __name__=='__main__':main()
