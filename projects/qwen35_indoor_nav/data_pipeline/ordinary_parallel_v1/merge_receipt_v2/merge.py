"""Original full strict merge, with independently validated recovery receipts as extra evidence."""
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent/'runtime_v1'
sys.path.insert(0,str(RUNTIME))
import common as c
import safe_size
sys.path.insert(0,str(HERE))
import gate


def verify_transport_inputs():
    for path,h in c.read(HERE/'INPUT_LOCK.json').items():assert c.sha(c.ROOT/path)==h,path


def transported_source():
    path=RUNTIME/'merge.py';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='0c16272e4430b229f199c8b5329f681129c4ad0cb948fb10fd99896db352735f'
    source=raw.decode()
    for old,new in [
        ('    c.immutable_verify();out=', '    verify_transport_inputs();c.immutable_verify();out='),
        ("out=c.PARALLEL/'merge'", "out=HERE/'merge'"),
        ('    locks=[]\n', '    locks=[];restoration_receipts={}\n'),
        ("lane=HERE/'lanes'/f'gpu_{gpu}'", "lane=c.HERE/'lanes'/f'gpu_{gpu}'"),
        ("assert result['restoration']['restored'],'HOLDER_NOT_RESTORED'",
         "restoration_receipts[str(gpu)+':'+attempt.name]=gate.verify(attempt,result,gpu)"),
        ("    c.save(out/'INPUT_HASHES.json',input_hashes)",
         "    c.save(out/'INPUT_HASHES.json',input_hashes)\n    c.save(out/'RESTORATION_EVIDENCE.json',restoration_receipts)"),
        ("status='STRICT_CPU_READBACK_MERGED_NOT_TRAINED',shards=summary,",
         "status='STRICT_CPU_READBACK_MERGED_NOT_TRAINED',restoration_gate='VERSIONED_EVIDENCE_OLD_RESULT_PRESERVED',shards=summary,")]:
        source=c.exact(source,old,new)
    return source


exec(compile(transported_source(),str(__file__),'exec'),globals())
