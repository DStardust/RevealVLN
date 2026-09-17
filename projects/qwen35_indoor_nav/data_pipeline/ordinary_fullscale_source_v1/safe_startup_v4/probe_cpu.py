"""Isolated project-socket tmux CPU probe; never connects to existing tmux/GPU jobs."""
import collections
import json
from pathlib import Path
import subprocess
import tempfile
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def identity(pid):
    p=Path('/proc',str(pid));fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime_ticks=int(fields[19]),uid=p.stat().st_uid,
        cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))


def main():
    assert not (HERE/'CPU_PROBE_RESULT.json').exists()
    records=[];server_started=False
    with tempfile.TemporaryDirectory(prefix='isolated_tmux_',dir=HERE) as folder:
        socket=Path(folder)/'server.sock'
        def call(*args):
            return subprocess.check_output(['tmux','-S',str(socket),*args],text=True,timeout=10).strip()
        try:
            call('-f','/dev/null','new-session','-d','-s','cpu_probe','-c',str(ROOT),'sleep 120')
            server_started=True
            shell=call('display-message','-p','-t','cpu_probe:0.0','#{default-shell}')
            for style,args in [('shell_string',['sleep 2']),('direct_argv',['/usr/bin/sleep','2'])]:
                for i in range(20):
                    out=call('new-window','-P','-F','#{pane_pid} #{pane_id}','-d','-t','cpu_probe','-c',str(ROOT),*args)
                    pid,pane=out.split();samples=[]
                    for _ in range(4):
                        try:samples.append(identity(int(pid)))
                        except FileNotFoundError:samples.append(dict(pid=int(pid),not_yet_present=True))
                        time.sleep(.002)
                    records.append(dict(style=style,launch_args=args,pane_id=pane,samples=samples))
        finally:
            if server_started:call('kill-server')  # exact private socket created above; no other server
    counts=collections.Counter((r['style'],tuple(s.get('argv',[]))) for r in records for s in r['samples'])
    result=dict(status='ISOLATED_CPU_TMUX_PROBE',default_shell=shell,records=records,
        observed=[dict(style=style,argv=list(argv),count=n) for (style,argv),n in counts.items()],
        existing_tmux_contacted=False,gpu_operations=0,existing_processes_signalled=0,
        private_cpu_server_cleaned=True)
    with (HERE/'CPU_PROBE_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['observed'],indent=2))


if __name__=='__main__':main()
