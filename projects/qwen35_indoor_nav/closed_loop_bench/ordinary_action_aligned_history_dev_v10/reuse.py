"""Same physics and metrics; aligned causal window and independent selected-RGB audit."""
import hashlib,importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_fp32_master_dev_v8r1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='334c646311cb43924ccfc71a013909a3a91ba1fb4eb5379ba60ccbc36bba0b8b'
s=importlib.util.spec_from_file_location('aligned_eval_parent',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def source(name):
    text=parent.source(name);changes={}
    if name=='common.py':
        changes={"Window=old.Window;advance=old.advance;xyzw_to_wxyz=old.xyzw_to_wxyz":"_history=load('aligned_history',LINE/'sft_acceptance/ordinary_action_aligned_history_v10/history_r1.py')\nWindow=_history.window_class(old.Window);advance=old.advance;xyzw_to_wxyz=old.xyzw_to_wxyz"}
    if name=='evaluate.py':
        changes={"executed_history=len(windows[lane].executed)))":"executed_history=len(windows[lane].executed),**windows[lane].input_audit()))"}
    if name=='aggregate.py':
        changes={
          "counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)":"counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)\n        observed={x['index']:[x['rgb_sha256']] for x in records(folder/'INTERFACE.jsonl')}",
          "                counters[index][step['action']]+=1;":"                t=st-1;selected=[0] if t==0 else [max(0,t-8),t]\n                assert len(observed[index])==st and policy['frame_stride']==8\n                assert policy['input_frame_indices']==selected\n                assert policy['input_rgb_sha256']==[observed[index][j] for j in selected]\n                observed[index].append(step['rgb_sha256'])\n                counters[index][step['action']]+=1;",
          "trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,":"trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,selected_history_audited_actions=audited,frame_stride=8,",
          "R2R-CE train INTERNAL_DEV; FP32 action master ordinary base":"R2R-CE train INTERNAL_DEV; action-aligned two-frame ordinary base"}
    for old,new in changes.items():assert text.count(old)==1,(name,old);text=text.replace(old,new)
    return text
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':aligned-window-audit','exec'),namespace)
