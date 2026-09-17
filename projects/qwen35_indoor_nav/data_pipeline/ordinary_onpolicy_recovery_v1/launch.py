"""Same bounded GPU1 empty-only owner cleanup; renderer-only worker and data audit."""
import hashlib,importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parents[2]/'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
s=importlib.util.spec_from_file_location('recovery_readonly_launch',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
text=m.source('launch.py')
changes={"'.envs/q35n_qwen_g2_v1/bin/python3'":"'.envs/q35n_habitat_v017_g0r/bin/python3'",
         "HERE/'evaluate.py'":"HERE/'worker.py'","HERE/'aggregate.py'":"HERE/'audit.py'"}
for old,new in changes.items():
    assert text.count(old)==1,old;text=text.replace(old,new)
exec(compile(text,__file__+':hash-bound-parent','exec'),globals())
