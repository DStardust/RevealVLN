import importlib.util as _iu
from pathlib import Path as _P
_s = _iu.spec_from_file_location('stop_dev_reuse', _P(__file__).with_name('reuse.py'))
_r = _iu.module_from_spec(_s)
_s.loader.exec_module(_r)
_r.execute('launch.py', globals())
