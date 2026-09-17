"""Official Marky-Matterport train annotations only; no synthetic relabeling."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
HERE=Path(__file__).resolve().parent
URL='https://storage.googleapis.com/rxr-data/rxr_marky_train_guide.jsonl.gz'
def execute():
    out=HERE/'marky_acquisition_v1';out.mkdir(exist_ok=False)
    target=out/'rxr_marky_train_guide.jsonl.gz';started=time.monotonic()
    # 512MiB cap + official archive <364MiB and two selected members remain <2GiB.
    result=subprocess.run(['curl','-sS','-L','--connect-timeout','20','--max-time','600',
        '--max-filesize',str(512*1024**2),'--output',str(target),URL],capture_output=True,text=True,timeout=610)
    digest=hashlib.sha256()
    if target.exists():
        with target.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2),b''):digest.update(chunk)
    receipt=dict(source='OFFICIAL_MARKY_MATTERPORT_GENERATED_INSTRUCTION_TRAIN',url=URL,
        official_reference='https://github.com/google-research-datasets/RxR/tree/main/marky-mT5',
        source_grade='OFFICIAL_MODEL_GENERATED_NOT_HUMAN_ANNOTATED',license='RxR CC-BY annotations plus Matterport3D scene terms',
        returncode=result.returncode,stderr=result.stderr[-1000:],bytes=target.stat().st_size if target.exists() else 0,
        sha256=digest.hexdigest(),wall_seconds=time.monotonic()-started,
        official_eval_content_read=False,training_allowed=False,gpu_operations=0)
    with (out/'RESULT.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)
    if result.returncode:raise SystemExit(1)
if __name__=='__main__':execute()
