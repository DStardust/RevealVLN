import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
parent=here/'launch_extract.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='2b1a6cdb226f95f7425a60742ae0d7c038c4a92a3655138d03eff9f9eb562d07'
assert json.loads((here/'LAUNCH_RESULT.json').read_text())['status']=='FAILED'
assert not (here/'FEATURES.pt').exists() and not (here/'EXTRACTION_RESULT.json').exists()
lock=json.loads((here/'TRANSPORT_R1_LOCK.json').read_text())
for path,digest in lock['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
text=parent.read_text()
old="env.update(CUDA_VISIBLE_DEVICES='1',";assert text.count(old)==1
text=text.replace(old,"env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='1',")
for old,new in [('LAUNCH_RESULT.json','LAUNCH_R1_RESULT.json'),('LAUNCH_FAILURE.json','LAUNCH_R1_FAILURE.json'),('PREFLIGHT.json','PREFLIGHT_R1.json'),('PROCESS.json','PROCESS_R1.json'),('RESOURCE.json','RESOURCE_R1.json'),('extract.log','extract_r1.log')]:
    assert old in text;text=text.replace(old,new)
exec(compile(text,str(Path(__file__))+':hash-bound-parent','exec'),globals())
