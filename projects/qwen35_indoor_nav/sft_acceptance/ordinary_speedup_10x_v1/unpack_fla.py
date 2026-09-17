"""Install the verified pure-Python wheel without modifying the pip-less env."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

HERE=Path(__file__).resolve().parent
folder=HERE/'official_fla_0_5_2'
wheel=folder/'fla_core-0.5.2-py3-none-any.whl'
expected='5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761'
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected
dest=folder/'deps';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(wheel) as z:
    assert 'Root-Is-Purelib: true' in z.read('fla_core-0.5.2.dist-info/WHEEL').decode()
    assert sum(x.file_size for x in z.infolist())<32*1024**2
    assert all(not Path(x.filename).is_absolute() and '..' not in Path(x.filename).parts and not any(p.endswith('.data') for p in Path(x.filename).parts) and (x.external_attr>>16)&0o170000!=0o120000 for x in z.infolist())
    for name,digest,size in csv.reader(io.StringIO(z.read('fla_core-0.5.2.dist-info/RECORD').decode())):
        if digest:
            algorithm,value=digest.split('=',1);assert algorithm=='sha256'
            data=z.read(name);assert len(data)==int(size)
            assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')==value
    z.extractall(dest)
manifest={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
with (folder/'PROVENANCE.json').open('x') as f:json.dump(dict(unix=time.time(),official_project='https://github.com/fla-org/flash-linear-attention',official_metadata='https://pypi.org/pypi/fla-core/0.5.2/json',sha256=expected,bytes=wheel.stat().st_size,version='0.5.2',installation='purelib wheel extraction; RECORD hashes verified; no pip/bootstrap in existing environment',prior_acquisition_issue='urllib TLS through registered proxy failed; curl with explicit registered proxy succeeded. Existing env has no pip; pip attempt changed nothing.',existing_environment_changed=False,files=manifest),f,indent=2)
print(json.dumps(dict(status='OFFICIAL_PURELIB_FLA_READY',files=len(manifest),existing_environment_changed=False)))
