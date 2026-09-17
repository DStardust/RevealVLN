import importlib.util,json
from pathlib import Path
_here=Path(__file__).resolve().parent
_p=_here.parents[1]/'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
_s=importlib.util.spec_from_file_location('prefix_parent_r1',_p);_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
_source=_m.source('common.py')
_old="TINY=HERE.parent/'r2r_ce_tiny_v1'";assert _source.count(_old)==1
_source=_source.replace(_old,"TINY=LINE/'closed_loop_bench/r2r_ce_tiny_v1'").replace("HERE/'SOURCE_LOCK.json'","HERE/'SOURCE_LOCK_R1.json'")
exec(compile(_source,str(Path(__file__))+':directory-transport-only','exec'),globals())
verify=verify_lock
def read(p):return json.loads(Path(p).read_text())
