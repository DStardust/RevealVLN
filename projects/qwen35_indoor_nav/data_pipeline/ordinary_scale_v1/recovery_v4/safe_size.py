"""Conservative apparent-byte census; only known atomic temporary ENOENT allowed."""
import os
from pathlib import Path
import re
import stat

PROJECT = Path('/mnt/data_nas/deeprobotics/daiyang/vla')

def known_temporary(path, root):
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) == 2 and parts[0] == 'content':
        return bool(re.fullmatch(r'[0-9a-f]{64}\.png\.tmp', parts[1]))
    if len(parts) == 2 and parts[0] == 'shards':
        return bool(re.fullmatch(r'shard_[0-9]{4}\.pending', parts[1]))
    if len(parts) == 2 and parts[0] == 'recovery_v4':
        return parts[1] in {'PROGRESS.json.pending', 'ASSET_LOCK_LIVE.json.pending',
                            'COUNTS_LAST_INVOCATION.json.pending', 'GENERATION_COMPLETE.json.pending'}
    return False

def measure(root, lstat=os.lstat):
    root = Path(root).absolute()
    if root.resolve(strict=True) != root or not root.is_relative_to(PROJECT):
        raise ValueError('ROOT_OUTSIDE_PROJECT_OR_SYMLINK')
    total = 0
    missing = []
    stack = [root]
    while stack:
        directory = stack.pop()
        total += lstat(directory).st_size
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    info = lstat(path)
                except FileNotFoundError:
                    if not known_temporary(path, root):
                        raise
                    missing.append(str(path.relative_to(root)))
                    # Avoid undercounting a promoted artifact absent from scandir's
                    # snapshot. Temporary RGB/JSON artifacts are much smaller;
                    # reserve 1 MiB per disappearance, plus the supervisor margin.
                    total += 1024 ** 2
                    continue
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError('SYMLINK_IN_PRODUCTION_TREE:' + str(path))
                if stat.S_ISDIR(info.st_mode):
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise ValueError('NON_REGULAR_PRODUCTION_ENTRY:' + str(path))
    return dict(apparent_bytes_conservative=total, tolerated_atomic_disappearances=missing)
