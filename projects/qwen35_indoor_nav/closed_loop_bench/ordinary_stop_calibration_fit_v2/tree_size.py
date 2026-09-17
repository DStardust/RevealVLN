"""Live output-size observation resilient only to concurrent ENOENT.

No symlink traversal, no suppression of permission/I/O errors. Counts transient
misses explicitly; as with all live scans this is an observation, not a snapshot.
"""
import os
from pathlib import Path
import stat


def tree_size(root):
    root=Path(root)
    if not root.is_dir():raise FileNotFoundError(root)
    todo=[root];size=misses=0
    while todo:
        folder=todo.pop()
        try:
            with os.scandir(folder) as stream:entries=list(stream)
        except FileNotFoundError:
            if folder==root:raise
            misses+=1;continue
        for entry in entries:
            try:
                value=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(value.st_mode):raise RuntimeError('OUTPUT_SYMLINK_FORBIDDEN:'+entry.path)
                if stat.S_ISDIR(value.st_mode):todo.append(Path(entry.path))
                elif stat.S_ISREG(value.st_mode):size+=value.st_size
            except FileNotFoundError:misses+=1
    return size,misses
