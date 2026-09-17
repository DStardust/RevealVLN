"""Exact V12 extractor on 32 fixed inputs, with only original FIT autotune cache."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;line=here.parents[1];parent=line/'sft_acceptance/ordinary_stop_row_v12'
source=(parent/'extract.py').read_text()
lock=json.loads((parent/'SOURCE_LOCK.json').read_text())
assert hashlib.sha256(source.encode()).hexdigest()==lock['files'][str(parent/'extract.py')]
source=source.replace("c.rows(HERE/'POLICY_INPUTS.jsonl')","c.rows(parent/'POLICY_INPUTS.jsonl')")
old='assert len(inputs)==5487';assert source.count(old)==1
source=source.replace(old,old+";inputs=[inputs[i] for i in list(range(16))+[i*300 for i in range(1,17)]]")
exec(compile(source,str(Path(__file__))+':hash-bound-extractor','exec'),globals())
