"""Versioned fixed4 transport; original variable-microbatch draft retained."""
from pathlib import Path
_source=Path(__file__).with_name('train.py').read_text()
for old,new in [("assert micro==8 and 32%micro==0","assert micro==4 and 32%micro==0"),
                ("HERE/'batching.py'","HERE/'batching_fixed4.py'")]:
    assert _source.count(old)==1
    _source=_source.replace(old,new)
exec(compile(_source,str(Path(__file__))+':fixed4','exec'),globals())

