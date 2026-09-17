"""Official source acquisition; proxy credentials never recorded."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
ENV=LINE/'.envs/q35n_qwen_g2_v1'
CACHE=LINE/'.cache/q35n_qwen_g2_v1'
MODEL=LINE/'runtime/models/Qwen3.5-2B_15852e8'
REV='15852e8c16360a2fea060d615a32b45270f8a8fc'


def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def get(url):
    with urllib.request.urlopen(url,timeout=45) as r:return r.read()


def environment():
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','CONDA_PREFIX','LD_LIBRARY_PATH','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','UV_INDEX','UV_DEFAULT_INDEX'):
        env.pop(k,None)
    CACHE.mkdir(parents=True,exist_ok=True)
    env.update(UV_CACHE_DIR=str(CACHE/'uv'),UV_PYTHON_INSTALL_DIR=str(CACHE/'python'),
        HF_HOME=str(CACHE/'hf'),XDG_CACHE_HOME=str(CACHE),TMPDIR=str(CACHE),
        PYTHONNOUSERSITE='1',UV_HTTP_TIMEOUT='90',UV_LINK_MODE='copy')
    return env


def setup():
    packages={'torch':'2.8.0','torchvision':'0.23.0','transformers':'5.15.0','peft':'0.18.0'}
    metadata={}
    for name,version in packages.items():
        data=json.loads(get(f'https://pypi.org/pypi/{name}/{version}/json'))
        metadata[name]={'version':data['info']['version'],'requires_python':data['info']['requires_python'],
                        'requires_dist':data['info']['requires_dist'],'source':f'https://pypi.org/pypi/{name}/{version}/json'}
    save('DEPENDENCY_METADATA.json',metadata)
    uv=str(ROOT/'.tools/uv/uv');py=str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3')
    env=environment()
    commands=[[uv,'venv','--python',py,str(ENV)],
        [uv,'pip','install','--python',str(ENV/'bin/python3'),'--index-url','https://pypi.org/simple',
         *[f'{k}=={v}' for k,v in packages.items()]]]
    for i,cmd in enumerate(commands):
        started=time.time()
        with (OUT/f'setup_{i}.log').open('x') as log:
            r=subprocess.run(cmd,env=env,cwd=LINE,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        save(f'setup_{i}_result.json',{'command':cmd,'returncode':r.returncode,'wall_seconds':time.time()-started})
        if r.returncode:raise RuntimeError(f'Setup {i} failed; see scoped log')
    r=subprocess.run([uv,'pip','freeze','--python',str(ENV/'bin/python3')],env=env,capture_output=True,text=True,check=True)
    with (OUT/'DEPENDENCIES_LOCK.txt').open('x') as f:f.write(r.stdout)
    print('Independent model environment installed',flush=True)


def model():
    api=f'https://huggingface.co/api/models/Qwen/Qwen3.5-2B/revision/{REV}?blobs=true'
    data=json.loads(get(api));save('MODEL_API_METADATA.json',data)
    assert data['sha']==REV
    names={'config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json',
           'tokenizer.json','chat_template.jinja','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json','README.md','LICENSE'}
    files=[r for r in data['siblings'] if r['rfilename'] in names or r['rfilename'].endswith('.safetensors') or r['rfilename'].endswith('.safetensors.index.json')]
    assert sum(r.get('size',0) for r in files)<6*1024**3
    MODEL.mkdir(parents=True,exist_ok=True)
    def download(r):
        name=r['rfilename'];assert '/' not in name
        p=MODEL/name;assert not p.exists()
        url=f'https://huggingface.co/Qwen/Qwen3.5-2B/resolve/{REV}/{name}'
        h=hashlib.sha256();n=0;started=time.time()
        with urllib.request.urlopen(url,timeout=90) as response,(MODEL/(name+'.partial')).open('xb') as f:
            while True:
                chunk=response.read(4*1024*1024)
                if not chunk:break
                n+=len(chunk);assert n<6*1024**3;h.update(chunk);f.write(chunk)
        if r.get('size') is not None:assert n==r['size'],name
        expected=r.get('lfs',{}).get('sha256')
        if expected:assert h.hexdigest()==expected,name
        (MODEL/(name+'.partial')).replace(p)
        result={'file':name,'bytes':n,'sha256':h.hexdigest(),'official_lfs_sha256':expected,'url':url,'wall_seconds':time.time()-started}
        print(json.dumps({'downloaded':name,'bytes':n}),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(download,files))
    save('MODEL_FILE_LOCK.json',results)
    print('Pinned official model files acquired',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    elif sys.argv[1]=='model':model()
