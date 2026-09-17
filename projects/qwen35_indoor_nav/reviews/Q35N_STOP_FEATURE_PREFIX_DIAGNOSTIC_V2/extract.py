"""Run only the immutable evaluation prefix; no simulator construction allowed."""
import importlib.util
from pathlib import Path
here=Path(__file__).resolve().parent;line=here.parents[1]
p=line/'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
s=importlib.util.spec_from_file_location('prefix_reuse',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
source=m.source('evaluate.py');marker='    windows=[c.Window()';assert source.count(marker)==1
source=source[:source.index(marker)]
marker='    # Before any new benchmark action:';assert source.count(marker)==1
insert="    diag=c.load('diagnostic_checks',HERE/'checks.py')\n    assert initial_sha==json.loads((c.LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001/MODEL_LOADED.json').read_text())['trainable_sha256']\n    early=diag.check(model,policy,collate,forward,c,'before')\n"
source=source.replace(marker,insert+marker)
source+="    late=diag.check(model,policy,collate,forward,c,'after')\n    assert fingerprint()==initial_sha\n    diag.finish(early,late,c,initial_sha)\n\nif __name__=='__main__':main()\n"
exec(compile(source,str(Path(__file__))+':original-evaluation-prefix','exec'),globals())
