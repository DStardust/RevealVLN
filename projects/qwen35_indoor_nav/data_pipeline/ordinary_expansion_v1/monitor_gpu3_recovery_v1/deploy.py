"""Exact reuse of verified monitor-only switch; account for natural lane closure."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'monitor_recovery_v1/deploy.py'
raw=SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='9ee60507ab17f34f54e9e0dd36b87eb94c5add51cd7bd74846bb173df9395b91'
source=raw.decode()
def exact(old,new,count=1):
    global source
    assert source.count(old)==count,(old,source.count(old),count)
    source=source.replace(old,new)

exact("OLD_ARGV=[PY,'-I','-S','-B',str(m.OLD/'monitor.py')]", "OLD_ARGV=[PY,'-I','-S','-B',str(HERE.parent/'monitor_recovery_v1/monitor.py')]")
exact('1421413','1530172',2)
exact("    cpu=m.read(HERE/'CPU_TESTS.json');js=m.read(HERE/'CLIENT_TESTS.json')\n    assert cpu['passed'] and cpu['tests']==8 and js['passed'] and js['tests']==8\n    for report,field in ((cpu,'tested_sha256'),(js,'source_sha256')):\n        for name,digest in report[field].items():assert m.sha(HERE/name)==digest,('TESTED_SOURCE_CHANGED',name)",
"    cpu=m.read(HERE/'CPU_TESTS.json')\n    assert cpu['passed'] and cpu['tests']==5\n    for name,digest in cpu['sha256'].items():assert m.sha(HERE/name)==digest,('TESTED_SOURCE_CHANGED',name)\n    previous=HERE.parent/'monitor_recovery_v1'\n    for filename,field in [('CPU_TESTS.json','tested_sha256'),('CLIENT_TESTS.json','source_sha256')]:\n        report=m.read(previous/filename);assert report['passed']\n        for name,digest in report[field].items():assert m.sha(previous/name)==digest,('INHERITED_MONITOR_CHANGED',name)")
exact("    protected=[identity(p) for p in (1421392,1421592,1421593)]", "    protected=protected_snapshot()")
exact("        for p in protected:assert identity(p['pid'])==p,'PROTECTED_PRODUCER_CHANGED'", "        verify_protected(protected)")
exact('def main():', '''def protected_snapshot():
    old=m.OLD
    entries=[(1421392,old/'launch_001/RESULT.json'),
             (1421592,old/'lanes/gpu_4/attempt_000/RESULT.json'),
             (1421593,old/'lanes/gpu_5/attempt_000/RESULT.json')]
    recovery=HERE.parent/'runtime_gpu3_recovery_v1'
    p=recovery/'launch_001/production_PROCESS.json'
    assert p.exists(),'RECOVERY_PROCESS_NOT_REGISTERED'
    entries.append((m.read(p)['pid'],recovery/'lanes/gpu_3/attempt_000/RESULT.json'))
    snapshots=[]
    for pid,result_path in entries:
        try:value=identity(pid)
        except FileNotFoundError:
            assert result_path.exists(),'UNACCOUNTED_PRODUCER_DISAPPEARANCE'
            value=None
        snapshots.append(dict(identity=value,result_path=str(result_path)))
    return snapshots

def verify_protected(snapshots):
    for entry in snapshots:
        old=entry['identity']
        if old is None:continue
        try:actual=identity(old['pid'])
        except FileNotFoundError:assert Path(entry['result_path']).exists(),'UNACCOUNTED_PRODUCER_EXIT'
        else:assert actual==old,'PROTECTED_PRODUCER_CHANGED'

def main():''')
exec(compile(source,__file__,'exec'),globals())
