"""Same frozen calibration rule; bind the transport-corrected FIT collection."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'fit.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='fe57d70a294a626afca66afc93e2608bb5c01447b57f9437fb55671de9792dc3'
text=raw.decode();old="CASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v1'"
assert text.count(old)==1
text=text.replace(old,"CASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'")
exec(compile(text,str(HERE/'fit_v2.py')+':unchanged-calibration','exec'),globals())
