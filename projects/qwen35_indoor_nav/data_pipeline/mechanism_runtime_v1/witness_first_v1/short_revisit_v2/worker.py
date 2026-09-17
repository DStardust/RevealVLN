"""Versioned worker: shorten only continuations, preserve sealed validators."""
import importlib.util
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def source_adapter(source):
    substitutions = [
        ("split='FIT'", "split='candidate_fit_pool'"),
        ("assert len(retained)==27", "assert len(retained)==27\n                    save(folder/'CERTIFICATE.json',certificate)"),
        ("save(folder/'CERTIFICATE.json',certificate);ledger.certify", "ledger.certify"),
        ("'kind':'witness_first_balanced_assembly_v1'", "'kind':'short_continuation_completed_subgoal_revisit_v2','control_type':cfg['control_type']"),
    ]
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source

def main():
    method = load('short_revisit_v2_method', HERE/'method.py')
    path = HERE.parent/'assembly_v1/worker.py'
    worker = types.ModuleType('short_revisit_v2_worker')
    worker.__file__ = str(path)
    exec(compile(source_adapter(path.read_text()), str(HERE/'worker.py'), 'exec'), worker.__dict__)
    worker.HERE = HERE
    worker.WitnessFactory = method.ShortRevisitFactory
    worker.main()

if __name__ == '__main__': main()
