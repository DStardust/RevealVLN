import hashlib,importlib.util,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
DATA=LINE/'data_pipeline/ordinary_route_teacher_v11/run_001'
BEST=LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt'
MODEL=LINE/'sft_acceptance/ordinary_sync_recovery_v1/model.py'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,data,exclusive=True):
    path=Path(path);tmp=path if exclusive else path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x' if exclusive else 'w') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    if not exclusive:tmp.replace(path)
def jsonl(path,data):
    with Path(path).open('x') as f:
        for x in data:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def verify():
    for p,h in read(HERE/'SOURCE_LOCK.json')['files'].items():assert sha(p)==h,'SOURCE_CHANGED:'+p
