"""Single exact output-path correction; old sealed audit and gates untouched."""
import argparse
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'audit.py'
ORIGINAL_SHA='6eef05c22c041abcd7040752897b1773ace0fac0a26a1af347b7d12191d546fc'
OLD="out=HERE/'audits'/batch.name"
NEW="out=GATE.parent/'auto_generation_v1'/batch.name"
def adapted_source(source):
    assert hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_SHA,'SEALED_AUTO_AUDIT_CHANGED'
    assert source.count(OLD)==1
    value=source.replace(OLD,NEW)
    assert value.count(NEW)==1 and value.replace(NEW,OLD)==source
    return value
private=types.ModuleType('auto_audit_output_scope_v2');private.__file__=str(ORIGINAL)
exec(compile(adapted_source(ORIGINAL.read_text()),str(ORIGINAL)+'::audit_output_scope_only','exec'),private.__dict__)
audit_main=private.audit_main
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);audit_main(p.parse_args().batch)
