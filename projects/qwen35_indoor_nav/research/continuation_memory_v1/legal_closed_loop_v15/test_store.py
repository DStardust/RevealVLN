"""CPU filesystem boundary check for the actual autonomous service store."""
import importlib.util
from pathlib import Path
import tempfile

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v15_service_store_test',root/'continuation_service.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(dir=root,prefix='store_cpu_') as folder:
    with module.AuditedContentStore(Path(folder)/'content',1024*1024) as store:
        item=store.put_raw(bytes([1,2,3]),'rgb',(1,1,3),'uint8')
        assert item['pixel_sha256']
        assert store.full_audit()['audit_pass']
try:module.AuditedContentStore(root.parent/'outside_v15_rejected_store',1024*1024)
except ValueError:pass
else:raise AssertionError('STORE_SCOPE_WIDENED')
module.c.write(root/'CONTINUATION_STORE_TEST_RESULT.json',dict(passed=True,
    actual_store_created_and_rehashed=True,outside_v15_rejected_before_creation=True,
    simulator_loaded=False,GPU_used=False))
