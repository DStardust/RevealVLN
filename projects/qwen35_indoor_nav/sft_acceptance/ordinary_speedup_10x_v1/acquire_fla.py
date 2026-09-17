"""Acquire a pinned official wheel, never change the running environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
URL='https://files.pythonhosted.org/packages/2d/ed/dfe19c4da779957eb6a42a26812f9b4e2280bf757a17a71933ff59ffcb98/fla_core-0.5.2-py3-none-any.whl'
SHA='5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761'

def main():
    auth=json.loads((HERE/'AUTHORIZATION.json').read_text())
    assert auth['approved'] and auth['kernel_acquisition']['existing_environment_changes_allowed'] is False
    folder=HERE/'official_fla_0_5_2';folder.mkdir(exist_ok=True)
    assert not any(folder.iterdir()), 'ONLY_EMPTY_PRE_DOWNLOAD_DIRECTORY_CAN_BE_REUSED'
    path=folder/'fla_core-0.5.2-py3-none-any.whl'
    # The registered authenticated proxy works with curl; urllib TLS failed
    # before downloading any artifact. Never print proxy credentials/argv.
    proxy=urllib.request.getproxies().get('https')
    assert proxy, 'REGISTERED_PROXY_REQUIRED'
    download=subprocess.run(['curl','--proxy',proxy,'--noproxy','','-fsSL','--max-time','30','--max-filesize','2097152',URL],capture_output=True,check=False)
    assert download.returncode==0, 'OFFICIAL_CURL_DOWNLOAD_FAILED_%d'%download.returncode
    blob=download.stdout
    assert len(blob)==819225 and hashlib.sha256(blob).hexdigest()==SHA
    with path.open('xb') as f:f.write(blob)
    with zipfile.ZipFile(path) as wheel:
        assert sum(i.file_size for i in wheel.infolist())<32*1024**2
        assert all(not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts for i in wheel.infolist())
        metadata=wheel.read('fla_core-0.5.2.dist-info/METADATA').decode()
    with (folder/'WHEEL_METADATA.txt').open('x') as f:f.write(metadata)
    env=os.environ.copy();env['TMPDIR']=str(folder/'tmp');Path(env['TMPDIR']).mkdir()
    result=subprocess.run([str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B','-m','pip','--isolated','install','--no-index','--no-deps','--no-cache-dir','--target',str(folder/'deps'),str(path)],env=env,check=True)
    manifest={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'deps').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    with (folder/'PROVENANCE.json').open('x') as f:json.dump(dict(unix=time.time(),url=URL,sha256=SHA,bytes=len(blob),version='0.5.2',pip_returncode=result.returncode,existing_environment_changed=False,files=manifest),f,indent=2)
    print(json.dumps(dict(status='ISOLATED_OFFICIAL_FLA_ACQUIRED',bytes=len(blob),files=len(manifest),existing_environment_changed=False)))

if __name__=='__main__':main()
