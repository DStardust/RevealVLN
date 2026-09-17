"""Read-only original-port monitor for SR40 paired ordinary-history training."""
import importlib.util
import json
from pathlib import Path
from http.server import ThreadingHTTPServer
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_sr40_v1/server.py'
s=importlib.util.spec_from_file_location('sr40_parent_monitor',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
TRAIN=HERE.parent/'ordinary_history8_paired_train_r1'
DIAG=LINE/'reviews/Q35N_HISTORY8_FIXED4_ACCUMULATION_V1'
def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return None
def live(identity):
    if not identity:return False
    root=Path('/proc')/str(identity['pid'])
    try:
        stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
        return stat[0] not in ('Z','X') and int(stat[19])==int(identity['start']) and root.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]==identity['argv'] and str(root.joinpath('cwd').resolve())==identity['cwd']
    except FileNotFoundError:return False
def collect(include_gpu=True):
    d=parent.collect(include_gpu=include_gpu);d['monitor_version']='ordinary_history8_pair_r1'
    arms={}
    for arm in ('control_recent2','treatment_prefix8'):
        root=TRAIN/arm;out=root/'run_001'
        process=read(root/'PROCESSES.json');ranks=(process or {}).get('ranks',[])
        history=[]
        try:
            for line in (out/'PROGRESS.jsonl').read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                history.append(dict(updates=row['updates'],ce=row['metrics']['mean_ce'],unix=row['unix']))
        except FileNotFoundError:pass
        arms[arm]=dict(progress=read(out/'PROGRESS.json'),result=read(out/'RESULT.json'),
            acceptance=read(root/'ACCEPTANCE.json'),launch=read(root/'LAUNCH_RESULT.json'),
            resource=read(root/'RESOURCE.json'),live_rank_pids=[x['pid'] for x in ranks if live(x)],history=history)
    d['paired_history']=dict(preparation=read(TRAIN.parent/'ordinary_history8_paired_train_v1/PREPARATION_RESULT.json'),protocol_present=(TRAIN/'PROTOCOL.json').is_file(),
        approval_present=(TRAIN/'MAIN_AGENT_APPROVAL.json').is_file(),arms=arms,
        diagnostic_result=read(DIAG/'RESULT.json'),diagnostic_launch=read(DIAG/'LAUNCH_RESULT.json'),
        diagnostic_process_live=live(read(DIAG/'PROCESS.json')),
        prior_batch_equivalence_result=read(LINE/'reviews/Q35N_HISTORY8_ACCUMULATION_INTERFACE_R1/RESULT.json'),
        lease=read(TRAIN/'lease_v1/LEASE_RESULT.json'),
        prior_memory_failure=read(TRAIN.parent/'ordinary_history8_paired_train_v1/control_recent2/LAUNCH_RESULT.json'),
        prior_memory_training=read(TRAIN.parent/'ordinary_history8_paired_train_v1/control_recent2/run_001/RESULT.json'),
        cpu_thread_test=read(TRAIN/'CPU_TEST_RESULT.json'))
    return d
class Handler(parent.Handler):
    def do_GET(self):
        try:
            if self.path=='/':self.respond(200,(HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
            elif self.path=='/api/status':self.respond(200,json.dumps(collect(),ensure_ascii=False,allow_nan=False).encode(),'application/json')
            elif self.path=='/healthz':self.respond(200,b'{"status":"ok","version":"ordinary_history8_pair_r1"}','application/json')
            else:self.respond(404,b'Not found','text/plain')
        except Exception as exc:self.respond(503,json.dumps(dict(error=repr(exc))).encode(),'application/json')
    do_HEAD=do_GET
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)


