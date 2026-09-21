"""Run registered commands under systemd, outside the Codex/terminal cgroup."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PYTHON=Path(os.environ.get('V16_STANDALONE_PYTHON',ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3')).resolve()
JOBS=HERE/'standalone_jobs'


def write(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(value,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)


def folder(name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,55}',name):raise ValueError('INVALID_JOB_NAME')
    return JOBS/name


def start(name,command):
    if command[:1]==['--']:command=command[1:]
    if not command or not Path(command[0]).is_absolute():raise ValueError('ABSOLUTE_EXECUTABLE_REQUIRED')
    executable=Path(command[0]).resolve()
    if not executable.is_file() or not (executable.is_relative_to(ROOT) or executable==PYTHON or executable==Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/.envs/q35n_habitat_v017_g0r/bin/python3').resolve()):raise ValueError('PROJECT_EXECUTABLE_REQUIRED')
    job=folder(name);job.mkdir(parents=True,exist_ok=False)
    unit='q35n-'+name+'.service'
    # Explicit numerical/runtime settings only; no API tokens or shell text.
    keys=('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
          'CUBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')
    spec=dict(name=name,unit=unit,command=command,cwd=str(ROOT),uid=os.getuid(),gid=os.getgid(),
        environment={k:os.environ[k] for k in keys if k in os.environ},created_unix=time.time(),
        launcher_pid=os.getpid(),launcher_cgroup=Path('/proc/self/cgroup').read_text(),
        retries=0,ownership='Same UID/GID as caller, no sudo or privilege change.')
    write(job/'JOB.json',spec)
    argv=['systemd-run','--unit='+unit,'--uid='+str(os.getuid()),'--gid='+str(os.getgid()),
        '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group',
        '--property=TimeoutStopSec=90','--property=RemainAfterExit=yes',
        str(PYTHON),'-I','-S','-B',str(Path(__file__).resolve()),'_worker',name]
    result=subprocess.run(argv,text=True,capture_output=True)
    write(job/'SUBMISSION.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr)
    print(json.dumps(dict(unit=unit,job=str(job),status_command=f'systemctl status {unit}',
        logs=str(job/'job.log'))))


def worker(name):
    job=folder(name);spec=json.loads((job/'JOB.json').read_text())
    assert os.getuid()==spec['uid'] and os.getgid()==spec['gid']
    def interrupted(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    began=time.time();state=dict(status='RUNNING',pid=os.getpid(),ppid=os.getppid(),started_unix=began,
        cgroup=Path('/proc/self/cgroup').read_text(),command=spec['command'],unit=spec['unit'])
    assert '/system.slice/'+spec['unit'] in state['cgroup'],'NOT_IN_INDEPENDENT_SERVICE'
    write(job/'STATUS.json',state)
    with (job/'job.log').open('x') as log:
        try:
            env=os.environ.copy();env.update(spec['environment']);env['PYTHONUNBUFFERED']='1'
            proc=subprocess.Popen(spec['command'],cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;write(job/'STATUS.json',state)
            code=proc.wait();state.update(status='COMPLETE' if code==0 else 'FAILED',returncode=code)
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=repr(error),returncode=1)
        finally:
            state.update(finished_unix=time.time(),wall_seconds=time.time()-began);write(job/'STATUS.json',state)
    raise SystemExit(state['returncode'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','status','_worker'))
    parser.add_argument('name');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.action=='start':start(args.name,args.command)
    elif args.action=='_worker':worker(args.name)
    else:
        job=folder(args.name);spec=json.loads((job/'JOB.json').read_text())
        print((job/'STATUS.json').read_text() if (job/'STATUS.json').exists() else 'NOT_STARTED')
        subprocess.run(['systemctl','show',spec['unit'],'-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],check=True)


if __name__=='__main__':main()
