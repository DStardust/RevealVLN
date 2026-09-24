"""Background, progress-reporting disk accounting; never block GPU health polling."""
import json,os,stat,subprocess,sys,time
from pathlib import Path

def charged_gpu_hours(prior,workers,now):
    return prior+sum(w.get('charged_hours',0) if w['done'] else (now-w['start'])/3600 for w in workers if w['gpu'])

def write(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def scan(root,output,progress_every=2):
    began=time.monotonic();last=0;pending=[Path(root)];seen=set();total=files=vanished=0
    def record(status):
        value=dict(status=status,unix=time.time(),bytes=total,files=files,vanished_during_scan=vanished,
                   seconds=time.monotonic()-began,semantics='Apparent bytes; hardlinks once; symlinks not followed. RUNNING bytes are a partial lower bound, not a complete measurement.')
        write(output,value);return value
    while pending:
        path=pending.pop()
        try:info=path.lstat()
        except FileNotFoundError:vanished+=1;continue
        key=info.st_dev,info.st_ino
        if key in seen:continue
        seen.add(key);total+=info.st_size
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as entries:pending.extend(Path(e.path) for e in entries)
            except FileNotFoundError:vanished+=1
        else:files+=1
        if time.monotonic()-last>=progress_every:record('RUNNING');last=time.monotonic()
    return record('COMPLETE')

class Monitor:
    def __init__(self,root,resources,python):
        self.root=Path(root);self.resources=resources;self.python=python;self.proc=None;self.last_start=-1e9;self.completed=None
        log=self.root/'DISK_MEASUREMENTS.jsonl'
        if log.exists():
            with log.open() as f:
                for row in f:
                    value=json.loads(row)
                    if value.get('status','COMPLETE')=='COMPLETE':self.completed=value
        self.folder=None;self.owner=None;self.log=None
    def poll(self):
        if self.proc is None and time.monotonic()-self.last_start>=300:
            self.folder=self.root/'disk_scans'/str(time.time_ns());self.folder.mkdir(parents=True)
            self.log=(self.folder/'worker.log').open('x');cmd=[self.python,'-I','-S','-B',str(Path(__file__).resolve()),str(self.root),str(self.folder/'PROGRESS.json')]
            self.proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            self.owner=self.resources.process_identity(self.proc.pid);self.last_start=time.monotonic()
            write(self.folder/'PROCESS.json',dict(owner=self.owner,command=cmd,role='read_only_disk_accounting',gpu=False))
        current=None
        if self.folder and (self.folder/'PROGRESS.json').exists():current=json.loads((self.folder/'PROGRESS.json').read_text())
        if self.proc is not None:
            code=self.proc.poll()
            if code is not None:
                write(self.folder/'EXIT.json',dict(returncode=code,wall_seconds=time.monotonic()-self.last_start,gpu_hours=0))
                self.log.close();self.log=None;self.proc=None
                if code or not current or current['status']!='COMPLETE':raise RuntimeError('DISK_SCANNER_FAILED')
                self.completed=current
                with (self.root/'DISK_MEASUREMENTS.jsonl').open('a') as f:f.write(json.dumps(current)+'\n')
            elif time.monotonic()-self.last_start>1200:raise TimeoutError('ASYNC_DISK_SCAN_1200_SECONDS')
        known=max(self.completed['bytes'] if self.completed else 0,current['bytes'] if current else 0)
        fs=os.statvfs(self.root)
        if fs.f_bavail*fs.f_frsize<2*2**30:raise RuntimeError('FILESYSTEM_FREE_SPACE_BELOW_2GIB')
        return dict(known_apparent_bytes=known,current=current,last_complete=self.completed,filesystem_available_bytes=fs.f_bavail*fs.f_frsize)
    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:self.resources.cleanup(self.proc,self.owner)
            write(self.folder/'EXIT.json',dict(returncode=self.proc.returncode,gpu_hours=0,cleanup=True))
            self.proc=None
        if self.log:self.log.close();self.log=None

if __name__=='__main__':scan(Path(sys.argv[1]),Path(sys.argv[2]))
