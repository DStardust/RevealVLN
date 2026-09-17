"""Official train-only selective ZIP member download, <=2GiB aggregate network."""
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
HERE=Path(__file__).resolve().parent
URL='https://drive.usercontent.google.com/download?id=145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t&export=download&confirm=t'
LIMIT=2*1024**3
class RangeFile(io.RawIOBase):
    def __init__(self,url,log):
        self.url=url;self.log=log;self.pos=0;self.downloaded=0
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
            self.length=int(r.headers['Content-Length']);assert self.length<LIMIT
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
        assert 0<=pos<=self.length;self.pos=pos;return pos
    def read(self,size=-1):
        if size<0:size=self.length-self.pos
        size=min(size,self.length-self.pos)
        if not size:return b''
        start=self.pos;end=start+size-1
        assert self.downloaded+size<=LIMIT
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=120) as response:
            assert response.status==206,'SERVER_DID_NOT_HONOR_RANGE'
            assert response.headers['Content-Range']==f'bytes {start}-{end}/{self.length}'
            data=response.read(size+1);assert len(data)==size
        self.downloaded+=len(data);self.pos+=len(data)
        self.log.append(dict(start=start,end=end,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        return data
def execute():
    out=HERE/'acquisition_v1';out.mkdir(exist_ok=False);events=[];error=None;members=[];remote=None
    try:
        remote=RangeFile(URL,events)
        with zipfile.ZipFile(remote) as archive:
            # Directory names are inspected, but held-out/non-target data bodies are never fetched.
            for base in ('train_follower.json.gz','train_follower_gt.json.gz'):
                names=[n for n in archive.namelist() if n.endswith('/train/'+base)]
                assert len(names)==1,('OFFICIAL_TRAIN_MEMBER_NOT_UNIQUE',base)
                info=archive.getinfo(names[0]);assert info.file_size<512*1024**2
                body=archive.read(info);assert len(body)==info.file_size
                path=out/base
                with path.open('xb') as f:f.write(body)
                members.append(dict(member=names[0],bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),zip_crc32=info.CRC))
    except BaseException as exc:error=repr(exc)
    finally:
        receipt=dict(official_url=URL,official_reference='https://github.com/jacobkrantz/VLN-CE/blob/master/README.md',
            train_only=True,heldout_member_bodies_read=False,members=members,range_requests=events,
            download_bytes=remote.downloaded if remote else 0,error=error,scientific_pass=False,
            license='CC BY-NC-SA 3.0 US plus Matterport3D terms; official VLN-CE README')
        with (out/'RESULT.json').open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt),flush=True)
    if error:raise SystemExit(1)
if __name__=='__main__':execute()
