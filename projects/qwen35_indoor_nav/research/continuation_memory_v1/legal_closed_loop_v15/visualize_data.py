"""Lossless raw RGB evidence, with labels outside the images; no generated pixels."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]


def main():
    audit=json.loads((HERE/'raw_run_008_AUDIT.json').read_text())
    config=json.loads((LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json').read_text())
    folder=HERE/'raw_rgb_evidence';folder.mkdir(exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><title>V15 raw data evidence</title>',
        '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{width:224px;height:224px}small{display:block}</style>',
        '<h1>Raw SEE2 evidence and the identical final RGB window</h1>',
        '<p>Physical data audit only. Each PNG is a lossless format conversion of the recorded RGB array. These images are not model-performance evidence.</p>']
    for split in ('fit','check'):
        family=next(f for f in config['families'] if f['partition']==split)
        cert=next(f for f in audit['families'] if f['family_id']==family['family_id'])
        parts.append('<h2>'+html.escape(split+' / '+family['house']+' / '+family['family_id'])+'</h2>')
        parts.append('<table><tr><th>History</th><th>Old event frame 1</th><th>Old event frame 2</th><th>Final window frame 1</th><th>Final window frame 2</th></tr>')
        for history,role in (('H_A','anchor_A'),('H_B','anchor_B')):
            witness=next(w for w in cert['visible_witnesses'] if w['history']==history and w['role']==role)
            run=LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008'
            trace=json.loads((run/family['family_id']/(history+'__C0.json')).read_text())
            cut=len(family['candidate']['histories'][history]);event=witness['first_step']
            label=history+' / '+family['roles'][role]['raw_match']['value']+' in '+family['roles'][role]['room']
            parts.append('<tr><th>'+html.escape(label)+'</th>')
            for step in (event-1,event,cut-1,cut):
                digest=trace['observations'][step]['rgb_hash'];pixels=np.load(run/'content'/(digest+'.rgb.npy'),allow_pickle=False)
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==digest
                target=folder/(digest+'.png')
                if not target.exists():Image.fromarray(pixels).save(target)
                parts.append(f'<td><img src="raw_rgb_evidence/{digest}.png"><small>observation {step}; SHA {digest[:12]}</small></td>')
            parts.append('</tr>')
        parts.append('</table><p>Same final raw RGB and executed-action windows; exact physical pose, allowing only the q/-q representation of one rotation. No interior state assignment.</p>')
    (HERE/'RAW_DATA_EVIDENCE.html').write_text('\n'.join(parts))
    print(HERE/'RAW_DATA_EVIDENCE.html')


if __name__=='__main__':main()
