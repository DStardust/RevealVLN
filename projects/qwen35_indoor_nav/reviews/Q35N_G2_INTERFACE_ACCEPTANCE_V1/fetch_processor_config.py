import hashlib
import json
from pathlib import Path
import urllib.request

OUT=Path(__file__).resolve().parent
MODEL=OUT.parents[1]/'runtime/models/Qwen3.5-2B_15852e8'
name='video_preprocessor_config.json'
meta=json.loads((OUT/'MODEL_API_METADATA.json').read_text())
r=next(r for r in meta['siblings'] if r['rfilename']==name)
url=f'https://huggingface.co/Qwen/Qwen3.5-2B/resolve/{meta["sha"]}/{name}'
with urllib.request.urlopen(url,timeout=45) as response:b=response.read()
assert len(b)==r['size']
with (MODEL/name).open('xb') as f:f.write(b)
with (OUT/'MODEL_SUPPLEMENT_LOCK.json').open('x') as f:
    json.dump([{'file':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'url':url,
                'reason':'Official shared processor declares a video component even though this probe only supplies images'}],f,indent=2)
print('Official processor component config acquired')
