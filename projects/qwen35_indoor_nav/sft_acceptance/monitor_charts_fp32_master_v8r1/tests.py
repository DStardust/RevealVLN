import runpy
from pathlib import Path
h=Path(__file__).resolve().parent;m=runpy.run_path(str(h/'server.py'))
d=m['collect'](include_gpu=False)
assert d['monitor_version']=='ordinary_fp32_master_v8r1'
assert d['onpolicy_training']['build']['recovery_unique_inputs']==4983
assert d['onpolicy_training']['stage_updates']==1000
assert 'navigation-results' in (h/'index.html').read_text()
assert d['navigation']['matched']['result']['sr']==.21
print('CPU_MONITOR_SCHEMA_AND_FIXED_BASE_PASS')

assert 'p.global_plan_decisions' in (h/'refresh.js').read_text()

assert d['onpolicy_training']['precision_bridge']['all_exec_and_query_bf16_injections_exact']

assert d['onpolicy_training']['transport']['status']=='PASS'
assert d['onpolicy_training']['prior_transport_failure']['status']=='FAILED_OR_BLOCKED'
assert 'id="transport"' in (h/'index.html').read_text()
assert 'id="evaluation"' in (h/'index.html').read_text()
