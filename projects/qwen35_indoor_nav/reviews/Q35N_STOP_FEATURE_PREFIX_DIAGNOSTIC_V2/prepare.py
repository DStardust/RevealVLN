import ast,hashlib,importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
fit=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2';parent=LINE/'sft_acceptance/ordinary_stop_row_v12'
files=dict(read(parent/'SOURCE_LOCK.json')['files'])
for name in ('PROTOCOL.json','PARITY_FIXTURES.json'):
    with (HERE/name).open('xb') as f:f.write((fit/name).read_bytes())
(HERE/'run_001').mkdir()
for p in (fit/'run_001/cache/triton').rglob('*.autotune.json'):
    target=HERE/'triton_cache'/p.relative_to(fit/'run_001/cache/triton');target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(p.read_bytes())
    files[str(p)]=sha(p)
for p in HERE.glob('*.py'):ast.parse(p.read_text())
# Compile the actual dynamically assembled prefix without running its GPU imports.
code=(HERE/'extract.py').read_text();code=code[:code.index('exec(compile(source,')]
ns={'__file__':str(HERE/'extract.py')};exec(compile(code,'CPU_PREFIX_ASSEMBLY','exec'),ns);ast.parse(ns['source'])
assert 'subprocess.Popen' not in ns['source'] and 'build_sim' not in ns['source'] and 'diag.finish' in ns['source']
write(HERE/'CPU_TEST_RESULT.json',dict(status='PASS',actual_prefix_ast=True,no_simulator_launch=True,source_model_cpu_tests=sha(parent/'CPU_TEST_RESULT.json'),gpu_forwards=0))
for p in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'PLAN_ZH.md',parent/'FEATURES.pt',fit/'run_001/MODEL_LOADED.json']:files[str(p)]=sha(p)
write(HERE/'SOURCE_LOCK.json',dict(files=files,unix=time.time()));print('EXACT_EVALUATOR_PREFIX_FROZEN')
