import importlib.util as _iu
from pathlib import Path as _P
_s=_iu.spec_from_file_location('expanded_dev_reuse',_P(__file__).resolve().parent.parent/'ordinary_expanded_dev_pair_v1/reuse_eval.py')
_r=_iu.module_from_spec(_s);_s.loader.exec_module(_r)
_r.execute('executor.py',globals())
