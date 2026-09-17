"""Same selected train members, curl TLS transport; prior urllib failure retained."""
from pathlib import Path
import subprocess
import io
HERE=Path(__file__).resolve().parent
class Response(io.BytesIO):
    def __init__(self,body,status,headers):super().__init__(body);self.status=status;self.headers=headers
def curl_open(request,timeout):
    args=['curl','-sS','-L','--proxy','http://127.0.0.1:27890','--suppress-connect-headers','--connect-timeout','15',
          '--max-time',str(timeout),'--max-filesize',str(2*1024**3),'-D','-']
    if request.get_method()=='HEAD':args+=['-I']
    for key,value in request.header_items():args+=['-H',key+': '+value]
    result=subprocess.run(args+[request.full_url],capture_output=True,timeout=timeout+5)
    assert result.returncode==0,('CURL',result.returncode,result.stderr.decode()[-1000:])
    raw=result.stdout;headers={};status=None
    while raw.startswith(b'HTTP/'):
        block,raw=raw.split(b'\r\n\r\n',1);lines=block.decode().splitlines();status=int(lines[0].split()[1])
        headers={k.strip().lower():v.strip() for k,v in (line.split(':',1) for line in lines[1:] if ':' in line)}
    return Response(raw,status,{k:headers[k.lower()] for k in ('Content-Length','Content-Range') if k.lower() in headers})
code=(HERE/'acquire.py').read_text()
assert code.count("out=HERE/'acquisition_v1'")==1
code=code.replace("out=HERE/'acquisition_v1'","out=HERE/'acquisition_v2'")
assert code.count('urllib.request.urlopen')==2
code=code.replace('urllib.request.urlopen','curl_open')
exec(compile(code,str(__file__),'exec'),globals())
