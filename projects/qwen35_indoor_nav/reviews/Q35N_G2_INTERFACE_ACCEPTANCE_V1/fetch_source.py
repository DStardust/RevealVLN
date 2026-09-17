import hashlib
import json
from pathlib import Path
import urllib.request

OUT=Path(__file__).resolve().parent
dest=OUT/'official_source';dest.mkdir(exist_ok=True)
records=[]
for name in ['modeling_qwen3_5.py','processing_qwen3_5.py']:
    url='https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_5/'+name
    with urllib.request.urlopen(url,timeout=45) as r:b=r.read()
    with (dest/name).open('xb') as f:f.write(b)
    records.append({'name':name,'url':url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
with (OUT/'OFFICIAL_SOURCE_LOCK.json').open('x') as f:json.dump(records,f,indent=2)
print('Official fixed-tag Qwen source downloaded for interface inspection')
