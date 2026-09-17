from pathlib import Path
source=Path(__file__).with_name('launch_extract.py').read_text()
assert "HERE/'common.py'" in source and "HERE/'extract.py'" in source
source=source.replace("HERE/'common.py'","HERE/'common_r1.py'").replace("HERE/'extract.py'","HERE/'extract_r1.py'")
exec(compile(source,str(Path(__file__))+':transport-r1','exec'),globals())
