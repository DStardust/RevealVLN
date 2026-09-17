import runpy
from pathlib import Path
h=Path(__file__).resolve().parent;m=runpy.run_path(str(h/'server.py'));d=m['collect'](include_gpu=False)
assert d['monitor_version']=='ordinary_equal_merge_v7r1'
assert d['equal_merge']['alpha']==.5 and d['equal_merge']['parameter_updates']==0
assert d['navigation']['matched']['result']['sr']==.21
assert 'navigation-results' in (h/'index.html').read_text()
print('MERGE_MONITOR_SCHEMA_AND_BASELINE_PASS')
