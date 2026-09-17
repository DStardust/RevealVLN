"""Original mixed-data BC recipe, continuing the same single-epoch plan."""
import hashlib
import importlib.util
from pathlib import Path

PARENT=Path(__file__).resolve().parent.parent/'ordinary_expanded_continue_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='3efc412c9cf7bc06489a9b4c071215dc12e632b22a75e817a16e3f6ea49d2429'
s=importlib.util.spec_from_file_location('full_epoch_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    text=parent.source(name);changes={}
    if name=='train_filestore.py':
        changes={"'Q35N_ORDINARY_EXPANDED_CONTINUE_V2'":"'Q35N_ORDINARY_EXPANDED_FULL_EPOCH_V4'"}
    if name=='supervise_filestore.py':
        changes={'triton_ordinary_expanded_continue_v2':'triton_ordinary_expanded_full_epoch_v4',
                 'inductor_ordinary_expanded_continue_v2':'inductor_ordinary_expanded_full_epoch_v4'}
    for old,new in changes.items():
        assert text.count(old)==1,old;text=text.replace(old,new)
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
