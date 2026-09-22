"""Unmodified V5 physical actions/metrics; asset-root relocation only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
service=u.load('stop13_frozen_executor',u.V5/'executor.py')
service.c.ROOT=u.ASSET.parents[1]
service.c.V3=u.ASSET/'closed_loop_bench/ordinary_cycle_pair_gpu1_v3'
service.main()
