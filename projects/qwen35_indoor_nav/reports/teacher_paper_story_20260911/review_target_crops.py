"""Diagnostic crops only; nearest-neighbor enlargement adds no scene detail."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'new_image_review/FRAME_AUDIT.json').read_text())
for row in rows:
    image=Image.new('RGB',(1050,540),'white');draw=ImageDraw.Draw(image)
    draw.text((10,5),row['attempt_id'],fill='black')
    frames=[f for f in row['frames'] if f['target_bbox']]
    for i,f in enumerate(frames):
        rgb=Image.open(f['png']);a,b,c,d=f['target_bbox']
        crop=rgb.crop((max(0,a-5),max(0,b-5),min(224,c+6),min(224,d+6)))
        scale=min(4,220//max(crop.size))
        crop=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
        x=10+(i%3)*345;y=35+(i//3)*250
        image.paste(crop,(x,y))
        draw.text((x,y+221),f['label']+' '+str(f['target_bbox']),fill='black')
    image.save(HERE/'new_image_review'/(row['attempt_id']+'_targets.png'))
