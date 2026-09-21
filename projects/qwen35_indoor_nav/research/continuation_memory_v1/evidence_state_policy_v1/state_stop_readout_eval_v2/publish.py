"""Publish this version and bounded evidence; scene arrays and base weights stay local."""
import fcntl
import io
import hashlib
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
prior=load('holdout_publish_helpers',PARENT/'gpu_runtime_r1/publish_r2.py')

def main(run):
    cfg=read(run/'PROTOCOL.json');branch=cfg['publication_branch']
    with (V16/'b2_kl_repair_v1/PUBLISH.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=branch:raise ValueError('BRANCH_CHANGED')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('OTHER_STAGED_CHANGES')
        out=HERE/'snapshots'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out.mkdir(parents=True)
        entries=[];archive=out/'LOGS.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=4) as bundle:
            for parent,dirs,files in os.walk(run):
                dirs[:]=[d for d in dirs if d not in ('content','features','cache','__pycache__')]
                for name in sorted(files):
                    p=Path(parent)/name
                    if p.suffix not in prior.TEXT or '.tmp' in name:continue
                    value=prior.blob(p);rel=str(p.relative_to(LINE));info=tarfile.TarInfo(rel);info.size=len(value)
                    bundle.addfile(info,io.BytesIO(value));entries.append(dict(path=rel,sha256=hashlib.sha256(value).hexdigest(),bytes=len(value)))
        parts=[]
        if archive.stat().st_size>40*2**20:
            with archive.open('rb') as stream:
                while value:=stream.read(40*2**20):
                    path=out/f'LOGS.tar.gz.part{len(parts):03d}';path.write_bytes(value);parts.append(path)
            archive.unlink()
        else:parts=[archive]
        paths=[p for p in HERE.glob('*') if p.is_file() and p.suffix in prior.TEXT]
        paths+=parts

        for name in ('PROTOCOL.json','SOURCE_LOCK.json','DATA_AUDIT.json','DATA.json','MODEL_REUSE.json','LOCAL_GATE.json','EVALUATION_REGISTRY.json','RESULT.json','REPORT_ZH.md','ROLLOUTS.csv','STATUS.json','RESOURCES.jsonl'):
            p=run/name
            if p.exists():paths.append(p)
        write(out/'INDEX.json',dict(files=entries,parts=[dict(path=p.name,sha256=sha(p)) for p in parts],
            excluded='Licensed scene/RGB/semantic arrays, environments and base checkpoint stay on server. All six frozen heads remain at state_stop_readout_v1/runs/repair_001/train or original gpu_v1; MODEL_REUSE gives exact references and hashes. Raw arrays and feature caches stay local.'),True)
        paths.append(out/'INDEX.json')
        for p in paths:
            if p.suffix in prior.TEXT:prior.blob(p)
        paths=[str(p.relative_to(ROOT)) for p in paths]
        for i in range(0,len(paths),80):subprocess.run(['git','add','-f','--',*paths[i:i+80]],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Evaluate frozen STOP readout with corrected semantic diagnostics'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        helper=LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'
        proxy=shlex.join([cfg['standalone_python'],'-I','-S','-B',str(helper),'%h','%p'])
        ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=ssh.github.com','-p','443','-o','ProxyCommand='+proxy])
        env=dict(os.environ,GIT_SSH_COMMAND=ssh)
        def remote(args):return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=1800)
        pushed=remote(['git','push','-u','origin',branch])
        if pushed.returncode:raise RuntimeError('GITHUB_PUSH_FAILED:'+pushed.stderr[-1500:])
        check=remote(['git','ls-remote','--heads','origin','refs/heads/'+branch])
        if check.returncode or not check.stdout.split() or check.stdout.split()[0]!=commit:raise ValueError('REMOTE_SHA_NOT_VERIFIED')
        append(HERE/'PUSH_RECEIPTS.jsonl',dict(commit=commit,branch=branch,status='PUSHED_AND_VERIFIED',unix=time.time()))

if __name__=='__main__':main(Path(sys.argv[1]))
