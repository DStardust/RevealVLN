"""Same optimizer/model/data algorithm, new namespace and read-only index path."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'ordinary_expanded_v1'
PARENT=OLD/'reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='109953572581f3b04d0565f806c85c9225a6ec59699f236890174559ba16f3d8'
s=importlib.util.spec_from_file_location('continued_expanded_reuse',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES


def source(name):
    text=parent.source(name)
    replacements={}
    if name=='train_filestore.py':
        replacements={"'Q35N_ORDINARY_EXPANDED_V1'":"'Q35N_ORDINARY_EXPANDED_LOW_LR_V3'",
            "HERE / 'SAMPLE_INDEX.jsonl'":"HERE.parent / 'ordinary_expanded_v1/SAMPLE_INDEX.jsonl'"}
    if name=='supervise_filestore.py':
        replacements={'triton_ordinary_expanded_v1':'triton_ordinary_expanded_low_lr_v3',
                      'inductor_ordinary_expanded_v1':'inductor_ordinary_expanded_low_lr_v3'}
    for old,new in replacements.items():
        assert text.count(old)==1,'SOURCE_ANCHOR_CHANGED:'+old
        text=text.replace(old,new)
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
