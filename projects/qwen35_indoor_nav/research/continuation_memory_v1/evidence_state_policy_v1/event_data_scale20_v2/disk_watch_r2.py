"""Non-blocking, owned disk telemetry; a timeout is unknown usage, not an exceeded limit."""
import os
import subprocess
import time
from pathlib import Path

class DiskWatch:
    def __init__(self,run,monitor,write,append,limit_bytes,interval=300,timeout=120,command=None):
        self.run=Path(run);self.monitor=monitor;self.write=write;self.append=append
        self.limit=limit_bytes;self.interval=interval;self.timeout=timeout
        self.command=command or ['du','-sb',str(run)]
        self.proc=None;self.last_start=float('-inf');self.last_success=None;self.log=None;self.owner=None
        old=self.run/'DISK.json'
        if old.exists():
            import json
            x=json.loads(old.read_text())
            if x.get('bytes') is not None:self.last_success=dict(bytes=x['bytes'],unix=x.get('measured_unix') or x['unix'])
    def snapshot(self,status,**extra):
        value=dict(status=status,bytes=self.last_success['bytes'] if self.last_success else None,
                   measured_unix=self.last_success['unix'] if self.last_success else None,
                   cap_bytes=self.limit,unix=time.time(),stale=status!='MEASURED',
                   per_house_writer_byte_cap_still_enforced=True,**extra)
        self.write(self.run/'DISK.json',value)
        self.append(self.run/'DISK_EVENTS_R2.jsonl',value)
    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:self.monitor.cleanup(self.proc,self.owner)
            if self.log is not None:self.log.close()
            self.proc=None;self.log=None
    def tick(self):
        now=time.monotonic()
        if self.proc is not None:
            code=self.proc.poll()
            if code is not None:
                self.log.close()
                text=self.path.read_text(errors='replace').strip()
                size=None
                if code==0:
                    try:size=int(text.split()[0])
                    except (ValueError,IndexError):pass
                self.proc=None;self.log=None
                if size is None:self.snapshot('SCAN_FAILED',returncode=code,log=str(self.path))
                else:
                    self.last_success=dict(bytes=size,unix=time.time())
                    self.snapshot('MEASURED',log=str(self.path))
                    if size>self.limit:raise RuntimeError('ARTIFACT_LIMIT')
            elif now-self.last_start>self.timeout:
                self.stop();self.snapshot('SCAN_TIMEOUT',log=str(self.path),timeout_seconds=self.timeout)
        if self.proc is None and now-self.last_start>=self.interval:
            folder=self.run/'disk_samples_r2';folder.mkdir(exist_ok=True)
            self.path=folder/(str(time.time_ns())+'.log');self.log=self.path.open('x')
            try:
                self.proc=subprocess.Popen(self.command,stdin=subprocess.DEVNULL,stdout=self.log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                self.owner=self.monitor.process_identity(self.proc.pid)
            except BaseException:
                self.log.close();raise
            self.last_start=now;self.snapshot('SCANNING',pid=self.proc.pid,log=str(self.path))

