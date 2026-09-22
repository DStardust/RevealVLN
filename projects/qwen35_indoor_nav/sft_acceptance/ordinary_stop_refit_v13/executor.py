"""Unmodified V5 physical actions/metrics; asset-root relocation only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
source=u.ASSET/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/executor.py'
assert u.sha(source)==u.sha(u.V5/'executor.py'),'EXECUTOR_SOURCE_NOT_IDENTICAL'
# The exact same source at its registered asset root resolves Habitat-Lab's
# geodesic_distance extraction and scene files without rewriting metric code.
service=u.load('stop13_frozen_executor',source)
service.main()
