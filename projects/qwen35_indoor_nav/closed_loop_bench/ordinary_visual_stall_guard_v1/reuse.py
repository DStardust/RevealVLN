"""Isolate one causal controller; preserve matched inference and physical audit."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_expanded_dev_after_single_v1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='e51c014048a6c189cbda790ff6e3c3608fc00ce3e437b2b269a325fc5a04038a'
s=importlib.util.spec_from_file_location('guard_matched_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    result=parent.source(name)
    replacements={}
    if name=='evaluate.py':
        replacements={
            "    windows=[c.Window() for _ in range(p['lanes'])]":
            "    guard_module=c.load('causal_visual_stall_guard',HERE/'guard.py')\n    guards=[guard_module.VisualStallGuard() for _ in range(p['lanes'])]\n    windows=[c.Window() for _ in range(p['lanes'])]",
            "        episode_steps[lane]=0":
            "        guards[lane].observe_rgb(windows[lane].images[-1])\n        episode_steps[lane]=0",
            "                    action=c.ACTIONS[max(range(4),key=values.__getitem__)]":
            "                    proposal=c.ACTIONS[max(range(4),key=values.__getitem__)]\n                    guard_receipt=guards[lane].choose(proposal)\n                    action=guard_receipt['action']",
            'action=action,logits=values,inference_batch=batches,images=len(windows[lane].images),executed_history=len(windows[lane].executed)))':
            'action=action,logits=values,model_proposed_action=proposal,guard=guard_receipt,inference_batch=batches,images=len(windows[lane].images),executed_history=len(windows[lane].executed)))',
            "                    alive=windows[lane].receive(call(lane,dict(op='action',action=action)),executed=action)":
            "                    alive=windows[lane].receive(call(lane,dict(op='action',action=action)),executed=action)\n                    if alive:guards[lane].observe_rgb(windows[lane].images[-1],executed=action)",
        }
    if name=='aggregate.py':
        replacements={
            '    overall=stats(rows);by_house=':
            "    guard_audit=c.load('independent_guard_audit',HERE/'guard_audit.py').audit(RUN,p)\n    overall=stats(rows);by_house=",
            'optimizer_updates=0,scientific_gain_verified=False,wall_seconds=launch':
            "optimizer_updates=0,scientific_gain_verified=False,pure_model_result=False,policy_controller='causal_visual_stall_guard_v1',guard_audit=guard_audit,wall_seconds=launch",
        }
    for old,new in replacements.items():
        assert result.count(old)==1, 'SOURCE_ANCHOR_CHANGED:'+old
        result=result.replace(old,new)
    return result


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':parent','exec'),namespace)
