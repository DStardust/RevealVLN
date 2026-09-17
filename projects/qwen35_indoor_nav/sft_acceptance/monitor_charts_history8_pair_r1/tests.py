import ast,json,runpy,shutil,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;m=runpy.run_path(str(h/'server.py'));d=m['collect'](False)
assert d['monitor_version']=='ordinary_history8_pair_r1'
assert d['paired_history']['preparation']['planned_decisions_per_arm']==384000
assert d['sr40']['official_episode_count']==1839 and d['sr40']['minimum_successes_for_40']==736
assert set(d['paired_history']['arms'])=={'control_recent2','treatment_prefix8'}
assert m['TRAIN']==m['LINE']/'sft_acceptance/ordinary_history8_paired_train_r1'
assert d['navigation']['matched']['result']['sr']==.21
assert d['onpolicy_training']['result']['after']['result']['sr']==.13
json.dumps(d,allow_nan=False)
for name in ('phase','diagnostic','training','training-time','loss-chart','navigation','resource','updated','error'):assert 'id="'+name+'"' in (h/'index.html').read_text()
for p in h.glob('*.py'):ast.parse(p.read_text())
subprocess.run([shutil.which('node'),'--check',str(h/'refresh.js')],check=True,capture_output=True)
print('PASS_MONITOR_DATA_AND_JS_SYNTAX; NO_BROWSER_VISUAL_TEST')


