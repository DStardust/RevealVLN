import ast,hashlib,importlib.util,json,runpy,time
from pathlib import Path
here=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
common=runpy.run_path(str(here/'common_r1.py'));assert common['TINY']==common['LINE']/'closed_loop_bench/r2r_ce_tiny_v1'
launch=runpy.run_path(str(here/'launch_extract_r1.py'),run_name='CPU_ONLY_IMPORT');assert callable(launch['main'])
assert launch['HERE']==here and launch['c'].TRAIN==common['LINE']/'sft_acceptance/ordinary_sync_recovery_v1'
for n in ('extract_r1.py','common_r1.py','launch_extract_r1.py'):ast.parse((here/n).read_text())
files=json.loads((here/'SOURCE_LOCK.json').read_text())['files']
for p in list(here.glob('*_r1.py'))+[here/'PRE_GPU_V1_FAILURE.json',here/'SOURCE_LOCK.json']:files[str(p)]=sha(p)
with (here/'SOURCE_LOCK_R1.json').open('x') as f:json.dump(dict(files=files,unix=time.time(),actual_launcher_and_common_cpu_import=True,zero_gpu_before_seal=True),f,indent=2)
common['verify']();assert not (here/'extract.log').exists();print('ACTUAL_CPU_IMPORT_AND_R1_LOCK_PASS')
