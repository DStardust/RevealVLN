"""Only the verified actual evaluation prefix; no simulator entry is assembled."""
import importlib.util
from pathlib import Path
here = Path(__file__).resolve().parent
line = here.parents[1]
path = line / 'closed_loop_bench/ordinary_stop_calibration_fit_v2/reuse.py'
spec = importlib.util.spec_from_file_location('accumulation_reuse', path)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
source = parent.source('evaluate.py')
marker = '    # Before any new benchmark action:'
assert source.count(marker) == 1
source = source[:source.index(marker)]
source += """    checks=c.load('accumulation_checks', HERE/'checks.py')
    checks.run(model,policy,collate,forward,c,fingerprint,initial_sha)
if __name__=='__main__':
    main()
"""
exec(compile(source, str(Path(__file__)) + ':actual-model-prefix', 'exec'), globals())



