"""Real upstream import and registered CPU head load; no GPU/base load."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('q35n_evaluator_import_test', HERE / 'evaluate_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
assert 'model' not in sys.modules
from runtime import LiveRuntime, ChunkActionProcessor, make_logits_processors, tensor_hash
from audit import audit_pair
from action_boundary_v2 import assistant_header
sys.path.remove(str(worker.HERE))
with tempfile.TemporaryDirectory(dir=HERE, prefix='import-test-') as tmp:
    worker.u.setup_imports(Path(tmp))
    import streamvln_eval as official
    import model
    protocol = worker.u.read(worker.HERE / 'runs/experiment_001/unseen/PROTOCOL.json')
    head = worker.load_registered_head(protocol['heads']['CURRENT_s42'], protocol, 'cpu')
    assert type(head) is worker.ExecutionAdaptation
    assert hasattr(model, '__path__')
    result = dict(status='REAL_IMPORT_AND_REGISTERED_HEAD_LOAD_PASS', unix=time.time(),
        upstream_package_paths=list(model.__path__), official_module=str(official.__file__),
        head_class_module=type(head).__module__, head_source_sha256=worker.u.sha(worker.HERE / 'model.py'),
        worker_sha256=worker.u.sha(HERE / 'evaluate_worker.py'), registered_head='CURRENT_s42',
        registered_head_sha256=protocol['heads']['CURRENT_s42']['sha256'],
        base_model_loaded=False, base_updates=0, optimizer_updates=0, gpu_hours=0)
    with (HERE / 'IMPORT_TEST_RESULT.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))
