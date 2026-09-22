"""Account live artifacts without failing on files atomically renamed by writers."""
import os,stat,time
from pathlib import Path
def measure(root,timeout=120):
    began=time.monotonic();root=Path(root);pending=[root];seen=set();total=0;vanished=0;files=0
    while pending:
        if time.monotonic()-began>timeout:raise TimeoutError('DISK_SCAN_DEADLINE')
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
    return dict(unix=time.time(),bytes=total,files=files,vanished_during_scan=vanished,seconds=time.monotonic()-began,semantics='apparent bytes; hard links counted once; symlinks not followed; only ENOENT is tolerated')
