"""Same checkpoint acceptance; new boundary 8000 -> 8200 only."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'Q35N_ORDINARY_CONTINUE_V2/accept_first.py'
raw=OLD.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f4ed7beedd37f0be010973bc01a004f933eb830354ee60d06ed38160c689aff4'
text=raw.decode()
for old,new in [('ordinary_expanded_continue_v2','ordinary_expanded_full_epoch_v4'),('4200','8200'),('4199','8199'),('4000','8000'),('340712','682651')]:
    assert old in text,old;text=text.replace(old,new)
exec(compile(text,str(HERE/'accept_first.py')+':unchanged-checks','exec'),globals())
