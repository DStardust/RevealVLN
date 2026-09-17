from pathlib import Path
source=Path(__file__).with_name('build.py').read_text()
old="assert c.read(c.DATA/'RESULT.json')['status'] in ('PASS','COMPLETE')"
assert source.count(old)==1
source=source.replace(old,"assert c.read(c.DATA/'RESULT.json')['status']=='AUDITED_ONPOLICY_TEACHER_DATA_READY' and c.read(c.DATA/'RESULT.json')['data_gate'] is True")
exec(compile(source,str(Path(__file__))+':versioned-status-correction','exec'),globals())
