from pathlib import Path
_source=Path(__file__).with_name('supervise.py').read_text()
assert _source.count("HERE/'train.py'")==1
_source=_source.replace("HERE/'train.py'","HERE/'train_fixed4.py'")
exec(compile(_source,str(Path(__file__))+':fixed4','exec'),globals())

