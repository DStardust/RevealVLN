"""Same frozen jobs; append only unattempted routes, preserve prior quarantine."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    preflight = load('recovery_v3_preflight', HERE / 'preflight.py')
    preflight.verify()
    worker = load('original_scale_worker', BASE / 'worker.py')
    worker.audit = load('unchanged_strict_quarantine', BASE / 'recovery_v1/audit.py')
    worker.audit.HERE = HERE
    def atomic(name, obj):
        target = HERE / name
        pending = target.with_suffix(target.suffix + '.pending')
        with pending.open('w') as handle:
            json.dump(obj, handle, indent=2, allow_nan=False)
        pending.replace(target)
    worker.atomic = atomic
    worker.main()

if __name__ == '__main__':
    main()
