"""Transport continuation only: same50 remaining updates in the identical plan."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_onpolicy_adapt_v6/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='7a1712c71453367140fa4b1f618c5e4ea1cc1b20d7e7874870e5c34245957e98'
s=importlib.util.spec_from_file_location('same_onpolicy_recipe',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES
def source(name):
    text=parent.source(name);changes={}
    if name=='train_filestore.py':
        changes={"'Q35N_ORDINARY_ONPOLICY_ADAPT_V6'":"'Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1'",
          "HERE / 'SAMPLE_INDEX.jsonl'":"HERE.parent / 'ordinary_onpolicy_adapt_v6/SAMPLE_INDEX.jsonl'"}
    if name=='supervise_filestore.py':
        changes={'triton_ordinary_onpolicy_adapt_v6':'triton_ordinary_onpolicy_adapt_v6r1','inductor_ordinary_onpolicy_adapt_v6':'inductor_ordinary_onpolicy_adapt_v6r1'}
    for old,new in changes.items():assert text.count(old)==1,old;text=text.replace(old,new)
    return text
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':unchanged-recipe','exec'),namespace)
