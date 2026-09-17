"""Read-only feature binding guard around frozen CPU fitting, no recipe changes."""
import hashlib,json,os,runpy,sys
from pathlib import Path
here=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sys.argv[1:] in (['probe'],['final'])
assert read(here/'LAUNCH_R1_RESULT.json')['status']=='COMPLETE'
assert read(here/'LAUNCH_R1_RESULT.json')['gpu_after']['processes']==[]
assert read(here/'FEATURE_PARITY.json')['status']=='PASS'
assert sha(here/'FEATURES.pt')==read(here/'FEATURE_PARITY.json')['feature_sha256']==read(here/'EXTRACTION_RESULT.json')['feature_sha256']
assert sha(here/'fit.py')==read(here/'SOURCE_LOCK.json')['files'][str(here/'fit.py')]
runpy.run_path(str(here/'fit.py'),run_name='__main__')
