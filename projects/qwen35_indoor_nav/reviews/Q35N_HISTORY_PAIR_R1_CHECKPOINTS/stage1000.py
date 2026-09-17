"""Read-only numerical audit at1000, using the frozen first200 checker with exact count changes."""
import hashlib
from pathlib import Path
import re
SOURCE=Path(__file__).with_name('first200.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='2973bbab466fa80b472859ba34a8b3f834412b0f873758df992c599f72101e0c'
code=SOURCE.read_text()
changes=[
 ("stage200","stage1000"),
 ("checkpoint_000000200.pt","checkpoint_000001000.pt"),
 ("PASS_FIRST200_NUMERICAL_ONLY","PASS_STAGE1000_NUMERICAL_ONLY"),
 ("_FIRST200.json","_STAGE1000.json"),
]
for old,new in changes:
    assert code.count(old)==1,old
    code=code.replace(old,new)
for old,new in [('200','1000'),('6400','32000'),('19200','96000')]:
    code,count=re.subn(r'\b'+old+r'\b',new,code)
    assert count>0,old
exec(compile(code,str(SOURCE)+':stage1000-counts-only','exec'),globals())

