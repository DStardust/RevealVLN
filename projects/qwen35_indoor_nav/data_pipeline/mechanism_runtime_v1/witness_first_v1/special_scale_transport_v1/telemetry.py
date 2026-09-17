"""Use reviewed ordinary accounting math with the original special guard label."""
import hashlib
import importlib.util
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('special_telemetry_common',HERE/'common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
ORIGINAL=c.LINE/'data_pipeline/ordinary_fullscale_source_v1/auto_generation_v4/telemetry.py'
c.require(c.sha(ORIGINAL)==c.SOURCES[ORIGINAL],'SEALED_ORDINARY_TELEMETRY')
source=ORIGINAL.read_text()
# The two guards use different exact assertion messages. No other catch broadening.
c.require(source.count('GPU_MEMORY_ACCOUNTING')==3,'GUARD_ERROR_LABEL_COUNT')
adapted=source.replace('GPU_MEMORY_ACCOUNTING','MEMORY_ACCOUNTING')
c.require(adapted.replace('MEMORY_ACCOUNTING','GPU_MEMORY_ACCOUNTING')==source,'TELEMETRY_REVERSE')
private=types.ModuleType('special_private_accounting');exec(compile(adapted,str(ORIGINAL)+'::special_guard_label','exec'),private.__dict__)

def assess(snapshot,own,guard,record):
    flat={'memory_mib':snapshot['memory_mib'],'processes':{p:v['mib'] for p,v in snapshot['processes'].items()}}
    # Validation/math sees original numeric values; the actual guard sees original
    # graphics inventory and remains the authority for all non-accounting errors.
    return private.active_assessment(flat,own,lambda unused,pid:guard(snapshot,pid),record)
