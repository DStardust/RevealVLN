"""Snapshot auditable production metadata; licensed arrays remain local."""
import fcntl
import hashlib
import io
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
prior=load('scale_publish_helpers',PARENT/'gpu_runtime_r1/publish_r2.py')

def main(run):
    cfg=read(run/'PROTOCOL.json')
    with (V16/'b2_kl_repair_v1/PUBLISH.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if branch!=cfg['publication_branch']:raise ValueError('BRANCH_CHANGED')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('OTHER_STAGED_CHANGES')
        out=HERE/'snapshots'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out.mkdir(parents=True)
        selected=[p for p in run.glob('*.json')]+[p for p in run.glob('*.md')]
        selected+=list((run/'collect').glob('*/position_*/FAMILY.json'))+list((run/'collect').glob('*/position_*/AUDIT.json'))
        selected+=list((run/'collect').glob('*/absent_*/FAMILY.json'))+list((run/'collect').glob('*/absent_*/AUDIT.json'))
        selected+=list((run/'collect').glob('*/ATTEMPTS.jsonl'))+list((run/'attempts').glob('*/EXIT.json'))
        entries=[]
        with tarfile.open(out/'METADATA.tar.gz','w:gz') as archive:
            for p in sorted(selected):
                value=prior.blob(p);name=str(p.relative_to(LINE))
                info=tarfile.TarInfo(name);info.size=len(value);archive.addfile(info,io.BytesIO(value))
                entries.append(dict(path=name,sha256=hashlib.sha256(value).hexdigest(),bytes=len(value)))
        for name in ('STATUS.json','COUNTS.json','REPORT_ZH.md'):
            p=run/name
            if p.exists():(out/name).write_bytes(prior.blob(p))
        immutable(out/'INDEX.json',dict(files=entries,archive_sha256=sha(out/'METADATA.tar.gz'),
            excluded='Licensed RGB/semantic arrays and full private traces remain local. Family certificates, trace SHA references and array SHA manifests included. No weights/training.'))
        paths=[p for p in HERE.iterdir() if p.is_file() and p.suffix in prior.TEXT]+list(out.iterdir())
        for p in paths:
            if p.suffix in prior.TEXT:prior.blob(p)
        subprocess.run(['git','add','-f','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Launch standalone 20x real SEE2 FIT data generation with eight-GPU monitoring'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        helper=LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'
        proxy=shlex.join([cfg['standalone_python'],'-I','-S','-B',str(helper),'%h','%p'])
        ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=ssh.github.com','-p','443','-o','ProxyCommand='+proxy])
        env=dict(os.environ,GIT_SSH_COMMAND=ssh)
        def remote(argv):
            return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(argv)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=480)
        for attempt in range(3):
            r=remote(['git','push','-u','origin',branch])
            if r.returncode==0:break
            append(HERE/'PUBLICATION.jsonl',dict(attempt=attempt,error=r.stderr[-1200:],commit=commit))
        if r.returncode:raise RuntimeError('PUSH_FAILED:'+r.stderr[-1200:])
        r=remote(['git','ls-remote','--heads','origin','refs/heads/'+branch])
        if r.returncode or not r.stdout.split() or r.stdout.split()[0]!=commit:raise ValueError('REMOTE_SHA_NOT_VERIFIED')
        append(HERE/'PUBLICATION.jsonl',dict(commit=commit,status='PUSHED_AND_VERIFIED',unix=time.time()))
        print(dict(commit=commit,verified=True),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))

