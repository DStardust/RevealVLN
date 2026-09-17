"""Frozen V8R1 learner; new route-target data index and namespace only."""
import hashlib,importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_onpolicy_fp32_master_v8r1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='433339071012c08f55a87f10336729f59bbb6daa88fa1247a877554fc3b2846d'
s=importlib.util.spec_from_file_location('route_recipe_parent',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES
def source(name):
    text=parent.source(name)
    return text.replace('Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1','Q35N_ORDINARY_ROUTE_TEACHER_V11').replace("HERE.parent / 'ordinary_onpolicy_adapt_v6/SAMPLE_INDEX.jsonl'","HERE / 'SAMPLE_INDEX.jsonl'").replace('triton_ordinary_fp32_master_v8r1','triton_ordinary_route_teacher_v11').replace('inductor_ordinary_fp32_master_v8r1','inductor_ordinary_route_teacher_v11')
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':route-teacher-data','exec'),namespace)
