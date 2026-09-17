import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
assert c.read(HERE/'CPU_TEST_RESULT.json')['status']=='PASS'
files=dict(c.read(c.FIT/'SOURCE_LOCK.json')['files']);files.update(c.read(HERE/'DATA_SOURCES.json')['files'])
for p in list(HERE.glob('*.py'))+[HERE/x for x in ('PLAN_ZH.md','POLICY_INPUTS.jsonl','SUPERVISION_ONLY.jsonl','SPLIT.json','BUILD_RESULT.json','CPU_TEST_RESULT.json','DATA_SOURCES.json')]+[c.MODEL,c.BEST]:files[str(p)]=c.sha(p)
for row in c.rows(HERE/'POLICY_INPUTS.jsonl'):
    for digest in row['rgb_sha256']:
        p=c.DATA/'content'/(digest+'.png')
        if str(p) not in files:files[str(p)]=c.sha(p)
for path,digest in files.items():assert c.sha(path)==digest,'SOURCE_CHANGED:'+path
c.write(HERE/'SOURCE_LOCK.json',dict(status='FROZEN',unix=time.time(),files=files));print(json.dumps(dict(files=len(files),bytes=sum(Path(x).stat().st_size for x in files),source_lock_sha256=c.sha(HERE/'SOURCE_LOCK.json'))))
