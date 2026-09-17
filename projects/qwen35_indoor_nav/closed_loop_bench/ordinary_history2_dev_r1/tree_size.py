from pathlib import Path
import runpy
_source=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v2/tree_size.py'
tree_size=runpy.run_path(str(_source))['tree_size']

