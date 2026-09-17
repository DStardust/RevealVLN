"""Exact frozen scout source adaptation, scoped to new_hub_scout_v1 and GPU2."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent;WF=HERE.parent;RUNTIME=WF.parent
spec=importlib.util.spec_from_file_location('newhub_frozen_bulk_adapter',WF/'bulk_source_v1/adapters.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
UUID='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
def exact(s,old,new):return a.exact(s,old,new)
def common_source():
    source=a.common_source(0)
    source=exact(source,repr(str(a.shard_root(0))),repr(str(HERE)))
    source=exact(source,'def select_hubs(backend,role_rows,positions,budget,target_function=candidate_targets):',
        'def _original_select_four_hubs(backend,role_rows,positions,budget,target_function=candidate_targets):')
    source=exact(source,'if len(selected)==2:break','if len(selected)==4:break')
    source+='\n_new_hub_selector=load("scoped_new_hub_selector",Path('+repr(str(HERE/'selector.py'))+'))\n'
    source+='def select_hubs(backend,role_rows,positions,budget,target_function=candidate_targets):\n'
    source+='    return _new_hub_selector.runtime_select(_original_select_four_hubs,backend,role_rows,positions,budget,runtime_config(),target_function)\n'
    return source
def worker_source():
    source=a.worker_source(0)
    source=exact(source,repr(str(a.shard_root(0))),repr(str(HERE)))
    source=exact(source,'from bulk_scout_scoped_common_00 import (','from new_hub_scoped_common_v1 import (')
    return exact(source,"HabitatBackend(candidate['scene_glb'],1,candidate['roles']","HabitatBackend(candidate['scene_glb'],2,candidate['roles']")
def supervisor_source():
    source=a.checked(RUNTIME/'compact_loop_v2/run.py')
    for old,new in [('HERE = Path(__file__).resolve().parent','HERE = Path('+repr(str(HERE))+')'),
        ('LINE = HERE.parents[2]','LINE = Path('+repr(str(a.ROOT/'projects/qwen35_indoor_nav'))+')'),
        ("'nvidia-smi','-i','1'","'nvidia-smi','-i','2'"),
        ("UUID = 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'",'UUID = '+repr(UUID))]:source=exact(source,old,new)
    return source
