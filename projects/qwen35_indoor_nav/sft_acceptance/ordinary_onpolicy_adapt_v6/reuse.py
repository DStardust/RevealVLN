"""Same ordinary model; explicit short new-data stage with original optimizer clock."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_expanded_continue_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='3efc412c9cf7bc06489a9b4c071215dc12e632b22a75e817a16e3f6ea49d2429'
s=importlib.util.spec_from_file_location('onpolicy_train_reuse',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES
def source(name):
    text=parent.source(name);changes={}
    if name=='train_filestore.py':
        changes={"'Q35N_ORDINARY_EXPANDED_CONTINUE_V2'":"'Q35N_ORDINARY_ONPOLICY_ADAPT_V6'",
          "HERE.parent / 'ordinary_expanded_v1/SAMPLE_INDEX.jsonl'":"HERE / 'SAMPLE_INDEX.jsonl'",
          "total_updates = sum(len(p[rank]) for p in plans)":"total_updates = protocol['lr_reference_total_updates']",
          "cosine_warmup(cursor['updates'], total_updates, warmup,":"cosine_warmup(cursor['updates'] + protocol['optimizer_step_offset'], total_updates, warmup,"}
    if name=='supervise_filestore.py':
        changes={'triton_ordinary_expanded_continue_v2':'triton_ordinary_onpolicy_adapt_v6',
                 'inductor_ordinary_expanded_continue_v2':'inductor_ordinary_onpolicy_adapt_v6'}
    for old,new in changes.items():
        assert text.count(old)==1,old;text=text.replace(old,new)
    return text
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
