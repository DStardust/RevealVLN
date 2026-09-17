"""Known atomic ENOENT only; permission/IO errors and symlinks fail closed."""
import importlib.util
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent/'ordinary_scale_v1'


def known(path,root):
    rel=path.relative_to(root);parts=rel.parts
    if len(parts)==1:
        return parts[0] in {'PROGRESS.json.pending','ASSET_LOCK_LIVE.json.pending',
            'COUNTS_LAST_INVOCATION.json.pending','GENERATION_COMPLETE.json.pending'}
    if len(parts)==2 and parts[0]=='content':return bool(re.fullmatch(r'[0-9a-f]{64}\.png\.tmp',parts[1]))
    if len(parts)==2 and parts[0]=='shards':return bool(re.fullmatch(r'shard_[0-9]{4}\.pending',parts[1]))
    return False


spec=importlib.util.spec_from_file_location('ordinary_parallel_safe_census',BASE/'recovery_v4/safe_size.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
original.known_temporary=known
measure=original.measure
