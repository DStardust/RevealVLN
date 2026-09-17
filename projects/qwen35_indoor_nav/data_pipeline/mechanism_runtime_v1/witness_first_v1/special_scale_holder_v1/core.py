"""Private GPU3/4/5/7 binding over sealed nonexclusive accounting/worker code."""
import importlib.util
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'special_scale_transport_v1'
BASE_SHA='bb3c5a41b3cb2f78a53cfc17ee860e75e7feb8389a859351ec5dd4ba49a7320a'
s=importlib.util.spec_from_file_location('holder_scale_common',BASE/'common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
GPUS={3:'GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a',4:'GPU-e458147b-4739-d22a-e764-50743cff4a11',
      5:'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b',7:'GPU-3f8830f6-45f2-6c4d-776b-2b348159ccb3'}

def adapted_source(source):
    c.require(c.sha(BASE/'transport.py')==BASE_SHA,'SEALED_SCALE_BASE_CHANGED')
    old="GPUS={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}"
    value=c.exact(source,old,'GPUS='+repr(GPUS))
    value=c.exact(value,"cfg['lease_mode']=='nonexclusive'","cfg['lease_mode']=='holder_chain'")
    value=c.exact(value,'    result=shared.supervisor_source(source,gpu)',
        '    shared.GPUS.update(GPUS)\n    result=shared.supervisor_source(source,gpu)')
    return value

def load_core():
    module=types.ModuleType('holder_private_scale_transport');module.__file__=str(BASE/'transport.py')
    exec(compile(adapted_source((BASE/'transport.py').read_text()),str(BASE/'transport.py')+'::holder_device_scope','exec'),module.__dict__)
    return module

private=load_core()
