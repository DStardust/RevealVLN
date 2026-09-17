"""Independent CPU decoding of every produced PNG; no simulator or GPU."""
import hashlib,json,os
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent;OUT=HERE/'run_001'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
count=0
for path in sorted((OUT/'content').glob('*.png')):
    with Image.open(path) as im:
        assert im.mode=='RGB' and im.size==(224,224)
        assert hashlib.sha256(im.tobytes()).hexdigest()==path.stem,'PNG_PIXEL_HASH'
    count+=1
with (OUT/'PIXEL_AUDIT.json').open('x') as f:json.dump(dict(passed=True,decoded_pngs=count),f,indent=2)
print('PIXEL_AUDIT_PASS',count)
