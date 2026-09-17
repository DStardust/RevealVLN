import importlib.util
import json
from pathlib import Path
_here = Path(__file__).resolve().parent
_path = _here.parents[1] / 'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
_spec = importlib.util.spec_from_file_location('history8_prefix_parent', _path)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)
_source = _parent.source('common.py')
_old = "TINY=HERE.parent/'r2r_ce_tiny_v1'"
assert _source.count(_old) == 1
_source = _source.replace(_old, "TINY=LINE/'closed_loop_bench/r2r_ce_tiny_v1'")
exec(compile(_source, str(Path(__file__)) + ':directory-only', 'exec'), globals())
def read(path):
    return json.loads(Path(path).read_text())


