"""Official CE EnvDrop split only, no validation/test member bodies fetched."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from acquire_v3 import RangeFile
URL='https://drive.usercontent.google.com/download?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr&export=download&confirm=t'
def execute():
    out=HERE/'envdrop_acquisition_v1';out.mkdir(exist_ok=False);events=[];members=[];error=None;remote=None
    previous=sum(json.loads(p.read_text()).get(k,0) for p,k in ((HERE/'acquisition_v3/RESULT.json','download_bytes'),
        (HERE/'marky_acquisition_v1/RESULT.json','bytes'),(HERE/'connectivity_acquisition_v1/RESULT.json','download_bytes')))
    try:
        remote=RangeFile(URL,events);assert remote.length+previous<2*1024**3,'AGGREGATE_DOWNLOAD_BUDGET'
        with zipfile.ZipFile(remote) as archive:
            for base in ('envdrop.json.gz','envdrop_gt.json.gz'):
                names=[n for n in archive.namelist() if n.endswith('/envdrop/'+base)]
                assert len(names)==1,('EXACT_ENVDROP_MEMBER_REQUIRED',base,names)
                info=archive.getinfo(names[0]);assert info.file_size<512*1024**2
                raw=archive.read(info);assert len(raw)==info.file_size
                with (out/base).open('xb') as f:f.write(raw)
                members.append(dict(member=names[0],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),zip_crc32=info.CRC))
    except BaseException as exc:error=repr(exc)
    finally:
        value=dict(official_url=URL,official_reference='https://jacobkrantz.github.io/vlnce/data',
            source_grade='OFFICIAL_ENVDROP_SYNTHETIC_INSTRUCTIONS_CE_PORT_NOT_HUMAN',
            archive_split='envdrop',training_augmentation_only=True,heldout_member_bodies_read=False,
            members=members,range_requests=events,download_bytes=remote.downloaded if remote else 0,
            prior_download_bytes=previous,aggregate_download_bytes=previous+(remote.downloaded if remote else 0),
            error=error,license='CC BY-NC-SA 3.0 US plus Matterport3D terms',training_allowed=False,gpu_operations=0)
        with (out/'RESULT.json').open('x') as f:json.dump(value,f,indent=2)
        print(json.dumps(value),flush=True)
    if error:raise SystemExit(1)
if __name__=='__main__':execute()
