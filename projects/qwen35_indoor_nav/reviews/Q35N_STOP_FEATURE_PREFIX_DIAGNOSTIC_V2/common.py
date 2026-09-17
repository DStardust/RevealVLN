import importlib.util,json
from pathlib import Path
_p=Path(__file__).resolve().parents[2]/'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
_s=importlib.util.spec_from_file_location('prefix_parent',_p);_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
_m.execute('common.py',globals())
verify=verify_lock
def read(p):return json.loads(Path(p).read_text())
