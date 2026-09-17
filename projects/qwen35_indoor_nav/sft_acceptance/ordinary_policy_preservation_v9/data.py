"""Read-only exact V6 mixed-data view and same1000update plan."""
import hashlib,runpy
from pathlib import Path
p=Path(__file__).resolve().parent.parent/'ordinary_onpolicy_adapt_v6/data.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='c11a85fe227d3243ee7e2cb6811ccde64b420c30a9ac136dac2541903f34029b'
m=runpy.run_path(str(p),run_name='FROZEN_V6_DATA')
for n in ('load_rows','load_sample_index','plan_epoch_batches','SampleStore','ACTIONS','advance_epoch_boundary'):globals()[n]=m[n]
