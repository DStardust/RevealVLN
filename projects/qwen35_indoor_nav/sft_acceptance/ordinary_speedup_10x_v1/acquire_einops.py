"""Pinned purelib dependency, official PyPI only, separate from frozen env."""
import base64,csv,hashlib,io,json,subprocess,time,urllib.request
from pathlib import Path
import zipfile
HERE=Path(__file__).resolve().parent
folder=HERE/'official_einops_0_8_1';folder.mkdir(exist_ok=False)
proxy=urllib.request.getproxies().get('https');assert proxy
def get(url,cap):
    assert url.startswith(('https://pypi.org/','https://files.pythonhosted.org/'))
    r=subprocess.run(['curl','--proxy',proxy,'--noproxy','','-fsSL','--max-time','30','--max-filesize',str(cap),url],capture_output=True,check=False)
    assert r.returncode==0,'OFFICIAL_DOWNLOAD_FAILED';assert len(r.stdout)<=cap;return r.stdout
metadata=get('https://pypi.org/pypi/einops/0.8.1/json',256*1024);m=json.loads(metadata)
item=next(x for x in m['urls'] if x['filename']=='einops-0.8.1-py3-none-any.whl')
blob=get(item['url'],1024**2);assert len(blob)==item['size'] and hashlib.sha256(blob).hexdigest()==item['digests']['sha256']
with (folder/item['filename']).open('xb') as f:f.write(blob)
with (folder/'OFFICIAL_METADATA.json').open('xb') as f:f.write(metadata)
dest=folder/'deps';dest.mkdir()
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    assert 'Root-Is-Purelib: true' in z.read('einops-0.8.1.dist-info/WHEEL').decode()
    assert sum(x.file_size for x in z.infolist())<8*1024**2
    assert all(not Path(x.filename).is_absolute() and '..' not in Path(x.filename).parts and (x.external_attr>>16)&0o170000!=0o120000 for x in z.infolist())
    for name,digest,size in csv.reader(io.StringIO(z.read('einops-0.8.1.dist-info/RECORD').decode())):
        if digest:
            alg,value=digest.split('=',1);data=z.read(name);assert alg=='sha256' and len(data)==int(size)
            assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')==value
    z.extractall(dest)
manifest={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
with (folder/'PROVENANCE.json').open('x') as f:json.dump(dict(unix=time.time(),version='0.8.1',url=item['url'],sha256=item['digests']['sha256'],bytes=len(blob),files=manifest,existing_environment_changed=False),f,indent=2)
print(json.dumps(dict(status='OFFICIAL_EINOPS_READY',bytes=len(blob),files=len(manifest))))
