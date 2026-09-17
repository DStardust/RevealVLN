"""Frozen numerical-correctness revision; same single FIT grid and gates."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
for path,expected in json.loads((HERE/'FIT_METRIC_REVISION_SEAL.json').read_text())['files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
raw=(HERE/'fit.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='fe57d70a294a626afca66afc93e2608bb5c01447b57f9437fb55671de9792dc3'
text=raw.decode()
changes={
 "CASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v1'":"CASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'",
 "s=importlib.util.spec_from_file_location('calibration_counterfactual',HERE/'counterfactual.py')":"s=importlib.util.spec_from_file_location('calibration_counterfactual',HERE/'counterfactual_v2.py')",
 "result=dict(status='COMPLETE',unix=time.time(),selected_bias=selected['bias'] if selected else None,":"result=dict(status='COMPLETE',unix=time.time(),metric_revision_seal_sha256=sha(HERE/'FIT_METRIC_REVISION_SEAL.json'),selected_bias=selected['bias'] if selected else None,",
}
for old,new in changes.items():
    assert text.count(old)==1,old;text=text.replace(old,new)
exec(compile(text,str(HERE/'fit_v3.py')+':official-float32-SPL','exec'),globals())
