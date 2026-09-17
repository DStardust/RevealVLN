"""Start the dashboard independently of the conversation/tool session."""
import os
from pathlib import Path
import subprocess
import sys
OUT=Path(__file__).resolve().parent
LIVE=OUT/'live';LIVE.mkdir(exist_ok=True)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TMPDIR']=str(OUT/'tmp')
with (LIVE/'server.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'-I','-B',str(OUT/'progress_server.py')],cwd=OUT.parents[3],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
