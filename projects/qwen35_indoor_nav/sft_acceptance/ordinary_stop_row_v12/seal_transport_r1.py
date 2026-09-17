import ast,hashlib,json,time
from pathlib import Path
here=Path(__file__).resolve().parent
paths=[here/x for x in ('launch_extract_r1.py','TRANSPORT_R1_ZH.md','SOURCE_LOCK.json','CPU_TEST_RESULT.json','LAUNCH_RESULT.json','extract.log','seal_transport_r1.py')]
ast.parse((here/'launch_extract_r1.py').read_text())
with (here/'TRANSPORT_R1_LOCK.json').open('x') as f:json.dump(dict(status='FROZEN_TRANSPORT_ONLY',unix=time.time(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),f,indent=2)
