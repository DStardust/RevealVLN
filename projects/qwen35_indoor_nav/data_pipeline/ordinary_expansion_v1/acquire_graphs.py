"""Pinned official MP3D connectivity for FIT scenes and official R2R train only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
COMMIT='589d091b111333f9e9f9d6cfd021b2eb68435925'
PREFIX='https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/'+COMMIT+'/'
def execute():
    out=HERE/'connectivity_acquisition_v1';out.mkdir(exist_ok=False)
    fit=json.loads((HERE.parent/'ordinary_fullscale_source_v1/SPLIT_FREEZE.json').read_text())['FIT']
    items=[('connectivity/'+s+'_connectivity.json',s+'_connectivity.json') for s in fit]
    items.append(('tasks/R2R/data/R2R_train.json','R2R_train.json'))
    def fetch(item):
        relative,name=item;target=out/name
        r=subprocess.run(['curl','-f','-sS','-L','--connect-timeout','15','--max-time','120','--max-filesize',str(8*1024**2),
            '--output',str(target),PREFIX+relative],capture_output=True,text=True,timeout=125)
        raw=target.read_bytes() if target.exists() else b''
        return dict(url=PREFIX+relative,filename=name,returncode=r.returncode,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),stderr=r.stderr[-500:])
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,items))
    value=dict(official_repository='peteanderson80/Matterport3DSimulator',revision=COMMIT,
        fit_houses=len(fit),requests=rows,download_bytes=sum(r['bytes'] for r in rows),
        success=all(r['returncode']==0 for r in rows),only_fit_connectivity_and_train_annotations=True,gpu_operations=0)
    with (out/'RESULT.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value),flush=True)
    if not value['success']:raise SystemExit(1)
if __name__=='__main__':execute()
