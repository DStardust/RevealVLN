"""Only observable kernel-cache configuration is recorded; no inferred dispatch."""
import json
import os
from pathlib import Path
from v16_common import HERE,sha,write

def record(destination):
    raw=os.environ.get('TRITON_CACHE_DIR');rows=[]
    if raw:
        root=Path(raw).resolve()
        if not root.is_relative_to(HERE):raise ValueError('TRITON_CACHE_OUTSIDE_V16')
        for p in sorted(root.rglob('*.json')):
            try:value=json.loads(p.read_text())
            except (OSError,ValueError):continue
            keys=('name','num_warps','num_ctas','num_stages','shared','cluster_dims','target','launch_cooperative_grid','enable_fp_fusion')
            if isinstance(value,dict) and any(k in value for k in keys):rows.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),configuration={k:value[k] for k in keys if k in value}))
    write(destination,dict(cache_directory=raw or 'unavailable',entries=rows,
        actual_complete_kernel_dispatch_sequence='unavailable',note='Compiled cache entries are observed configurations, not proof that every cached kernel executed.'))
