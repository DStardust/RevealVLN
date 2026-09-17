"""One FIT-frozen scalar applied to STOP only. Raw/adjusted logits both logged."""
import hashlib
import importlib.util
from pathlib import Path

PARENT=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
s=importlib.util.spec_from_file_location('stop_dev_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    text=parent.source(name)
    changes={}
    if name=='aggregate.py':
        changes={'R2R-CE train FIT threshold calibration; not validation':'R2R-CE train INTERNAL_DEV; single FIT-calibrated STOP offset'}
    elif name=='evaluate.py':
        changes={
          "OUT=HERE/'run_001'":"""OUT=HERE/'run_001'
_stop=c.load('fit_only_stop_rule',c.LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1/counterfactual.py')""",
          "                    action=c.ACTIONS[max(range(4),key=values.__getitem__)]":"""                    raw_values=list(values)
                    action=_stop.choose(raw_values,p['stop_logit_bias'])
                    values=list(raw_values);values[3]+=p['stop_logit_bias']
                    assert action==c.ACTIONS[max(range(4),key=values.__getitem__)]""",
          "action=action,logits=values,inference_batch=batches,images=len(windows[lane].images),executed_history=len(windows[lane].executed)))":"action=action,logits=values,raw_logits=raw_values,stop_logit_bias=p['stop_logit_bias'],inference_batch=batches,images=len(windows[lane].images),executed_history=len(windows[lane].executed)))",
        }
    for old,new in changes.items():
        assert text.count(old)==1,old;text=text.replace(old,new)
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
