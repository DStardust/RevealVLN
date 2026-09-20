"""Push this handoff using proxyon, then verify the exact remote branch commit."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BRANCH='codex/q35n-v15-legal-data-20260919'
PYTHON=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch==BRANCH,'UNEXPECTED_CHECKOUT'
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    proxy=shlex.join([str(PYTHON),'-I','-S','-B',str(HERE/'proxy_connect.py'),'%h','%p'])
    ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ProxyCommand='+proxy])
    env=dict(os.environ,GIT_SSH_COMMAND=ssh)
    def command(args,capture=False):
        # proxyon supplies credentials only in this child's environment.
        shell='proxyon >/dev/null 2>&1 && exec '+shlex.join(args)
        return subprocess.run(['bash','-lc',shell],cwd=ROOT,env=env,text=True,
            stdout=subprocess.PIPE if capture else None,check=True)
    began=time.time()
    command(['git','push','-u','origin',BRANCH])
    remote=command(['git','ls-remote','--heads','origin','refs/heads/'+BRANCH],True).stdout.strip()
    assert remote.split()[0]==local,'REMOTE_COMMIT_MISMATCH'
    result=dict(status='PUSHED_AND_REMOTE_VERIFIED',branch=BRANCH,commit=local,seconds=time.time()-began,
        branch_url='https://github.com/DStardust/RevealVLN/tree/'+BRANCH,
        readme_url='https://github.com/DStardust/RevealVLN/blob/'+local+'/'+str((HERE/'README_ZH.md').relative_to(ROOT)),
        credential_material_written=False)
    with (HERE/'PUSH_RESULT.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
