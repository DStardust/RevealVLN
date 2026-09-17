"""V6 scientific recipe unchanged except two FP32 master parameter groups."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_onpolicy_adapt_v6/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='7a1712c71453367140fa4b1f618c5e4ea1cc1b20d7e7874870e5c34245957e98'
s=importlib.util.spec_from_file_location('fp32_master_recipe',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES
def source(name):
    text=parent.source(name);changes={}
    if name=='train_filestore.py':
        changes={"'Q35N_ORDINARY_ONPOLICY_ADAPT_V6'":"'Q35N_ORDINARY_FP32_MASTER_V8'",
          "HERE / 'SAMPLE_INDEX.jsonl'":"HERE.parent / 'ordinary_onpolicy_adapt_v6/SAMPLE_INDEX.jsonl'",
          "        opt.load_state_dict(copy.deepcopy(state['optimizer']))":"        opt.load_state_dict(copy.deepcopy(state['optimizer']))\n        model.assert_master_precision(raw_policy,opt,state)\n        print(json.dumps(dict(event='MASTER_PRECISION_READY',rank=rank,all_trainable_float32=True,initial_bf16_injections_equal=True)),flush=True)"}
    if name=='supervise_filestore.py':
        changes={'triton_ordinary_onpolicy_adapt_v6':'triton_ordinary_fp32_master_v8','inductor_ordinary_onpolicy_adapt_v6':'inductor_ordinary_fp32_master_v8'}
    for old,new in changes.items():assert text.count(old)==1,old;text=text.replace(old,new)
    return text
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':one-precision-change','exec'),namespace)
