"""CPU source/data binding and JavaScript syntax. Not a browser visual test."""
import ast
import json
from pathlib import Path
import runpy
import shutil
import subprocess

here = Path(__file__).resolve().parent
module = runpy.run_path(str(here / 'server.py'))
data = module['collect'](False)
goal = data['sr40']
assert data['monitor_version'] == 'ordinary_sr40_v1'
assert module['REVIEW'] == module['LINE'] / 'reviews/Q35N_SR40_BASELINE_RESET_V1'
assert goal['stage'] == 'BASELINE_RESET_PREPARATION' and not goal['pass_goal']
assert goal['official_episode_count'] == 1839 and goal['minimum_successes_for_40'] == 736
assert goal['snapshot']['instruction_conditioned_decisions'] == 2650347
assert goal['historical_full']['sr'] == .22022838499184338
assert goal['historical_full']['selected_batch_size'] == 8
assert goal['best_internal']['sr'] == .21
assert data['navigation']['matched']['result']['sr'] == goal['best_internal']['sr']
assert data['onpolicy_training']['result']['after']['result']['sr'] == .13
assert data['sr40_training'] is None and data['sr40_full_result'] is None
assert goal['new_training_updates'] == 0 and not goal['prior_stop_fit_started']
assert goal['prefix_diagnostic']['numerical_32_pass']
assert goal['prefix_diagnostic']['launcher_failure_retained']
assert goal['prefix_diagnostic']['unknown_child_exit_code'] is None
json.dumps(data, allow_nan=False)
html = (here / 'index.html').read_text()
for element in ('phase','count','next','snapshot','official','gate','internal','resource','updated','error'):
    assert 'id="' + element + '"' in html
js = (here / 'refresh.js').read_text()
assert "fetch('/api/status'" in js and 'textContent' in js and 'innerHTML' not in js
for path in here.glob('*.py'):
    ast.parse(path.read_text())
node = shutil.which('node')
assert node, 'NODE_UNAVAILABLE'
result = subprocess.run([node, '--check', str(here / 'refresh.js')], capture_output=True, text=True, timeout=10)
assert result.returncode == 0, result.stderr
print('SR40_BINDING_AND_JS_SYNTAX_PASS; NO_BROWSER_VISUAL_TEST')

