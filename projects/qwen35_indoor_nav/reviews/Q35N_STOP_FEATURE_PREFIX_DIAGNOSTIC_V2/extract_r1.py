from pathlib import Path
source_text=Path(__file__).with_name('extract.py').read_text()
old="exec(compile(source,str(Path(__file__))+':original-evaluation-prefix','exec'),globals())"
assert source_text.count(old)==1
source_text=source_text.replace(old,"source=source.replace(\"HERE/'common.py'\",\"HERE/'common_r1.py'\")\n"+old)
exec(compile(source_text,str(Path(__file__))+':transport-r1','exec'),globals())
