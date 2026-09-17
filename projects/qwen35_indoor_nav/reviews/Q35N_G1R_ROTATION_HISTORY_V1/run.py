"""Use the preceding resource monitor unchanged, with isolated output/worker."""
import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
path = OUT.parent/'Q35N_G1R_COVERAGE_V1/run.py'
s = importlib.util.spec_from_file_location('bounded_runner', path)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.OUT = OUT
if __name__ == '__main__': m.main()
