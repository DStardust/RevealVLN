"""Read-only reuse of the tested live-directory scan correction."""
import hashlib
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v2/tree_size.py'
raw=PARENT.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a23c716d11983cd3786c753f15dc94369a19654432bcf4354b92acdfa018392c'
exec(compile(raw,str(PARENT)+':read-only','exec'),globals())
