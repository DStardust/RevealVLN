"""Frozen baseline evaluation with explicit history/model and audit adaptations only."""
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_stop_calibration_fit_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
s=importlib.util.spec_from_file_location('history_eval_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def source(name,arm):
    assert arm in ('control_recent2','treatment_prefix8')
    text=parent.source(name);changes=[]
    if name=='common.py':
        changes.append(("TRAIN=LINE/'sft_acceptance/ordinary_sync_recovery_v1'","TRAIN=LINE/'sft_acceptance/ordinary_history8_paired_train_r1'"))
        adapter=("_history=load('history_pair_window',LINE/'sft_acceptance/ordinary_prefix_history8_v1/history.py')\nWindow=_history.PrefixWindow"
                 if arm=='treatment_prefix8' else
                 "_history=load('history_pair_window',HERE.parent/'ordinary_history_pair_eval_r1/history_audit.py')\nWindow=_history.RecentWindow")
        changes.append(("Window=old.Window;advance=old.advance;xyzw_to_wxyz=old.xyzw_to_wxyz",
            adapter+";advance=old.advance;xyzw_to_wxyz=old.xyzw_to_wxyz"))
    elif name=='evaluate.py':
        changes.extend([
            ("assert state['cursor']['updates']==p['checkpoint_updates']",
             "assert state['cursor']['updates']==p['checkpoint_updates']\n    assert state['binding']['arm']==p['history_arm']"),
            ("executed_history=len(windows[lane].executed)))",
             "executed_history=len(windows[lane].executed),**windows[lane].input_audit()))")
        ])
    elif name=='aggregate.py':
        changes.extend([
            ("    rows=[];audited=0","    rows=[];audited=0\n    history_audit=c.load('independent_rgb_history_audit',HERE.parent/'ordinary_history_pair_eval_r1/history_audit.py')"),
            ("counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)",
             "counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)\n        observed={x['index']:[x['rgb_sha256']] for x in records(folder/'INTERFACE.jsonl')}"),
            ("assert policy['images']==min(2,st) and policy['executed_history']==min(8,st-1)",
             "history_audit.validate(p['history_arm'],policy,step,observed[index])\n                observed[index].append(step['rgb_sha256'])"),
            ("R2R-CE train FIT threshold calibration; not validation","R2R-CE train INTERNAL_DEV100; ordinary history pair R1"),
            ("trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,",
             "trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,selected_history_audited_actions=audited,history_arm=p['history_arm'],")
        ])
    elif name=='launch.py':
        old="    status=json.loads((c.TRAIN/'formal/STATUS.json').read_text())\n    return dict(status=status,processes=[identity(pid) for pid in (1151310,1151311,1151312,1151313)])"
        new="""    states={};processes=[]
    for arm in ('control_recent2','treatment_prefix8'):
        root=c.TRAIN/arm;path=root/'PROCESSES.json'
        ranks=json.loads(path.read_text())['ranks'] if path.exists() else []
        for rank in range(3):
            row=ranks[rank] if rank<len(ranks) else None
            now=identity(row['pid']) if row else None
            processes.append(now if now and now['start_ticks']==row['start'] else None)
        state=root/'run_001/RESULT.json'
        states[arm]=json.loads(state.read_text()) if state.exists() else None
    return dict(status=states,processes=processes)"""
        changes.append((old,new))
    for old,new in changes:
        assert text.count(old)==1,(name,old)
        text=text.replace(old,new)
    return text

