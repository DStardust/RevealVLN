import runpy
from pathlib import Path
h=Path(__file__).resolve().parent;m=runpy.run_path(str(h/'server.py'));d=m['collect'](False)
assert d['monitor_version']=='ordinary_stop_row_v12r1'
assert d['stop_row']['build']['unique_inputs']==5487 and not d['stop_row']['positive_navigation_result']
assert d['onpolicy_training']['result']['after']['result']['sr']==.13
assert d['navigation']['matched']['result']['sr']==.21
assert m['TRAIN']==m['LINE']/'sft_acceptance/ordinary_stop_row_v12'
for name in ('phase','progress','count','probe','results','verdict','resource','updated','error'):assert 'id="'+name+'"' in (h/'index.html').read_text()
assert 'ordinary_stop_row_dev_v12' in (h/'server.py').read_text()
print('CPU_MONITOR_BINDING_PASS; NO_BROWSER_VISUAL_TEST')
