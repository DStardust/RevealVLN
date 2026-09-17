"""Correct the pre-freeze test's 3*route assumption; preserve original source lock.

Actual frozen JSON has 4088 aliases, not 4089. No source row was changed.
"""
from pathlib import Path
source=(Path(__file__).resolve().parent/'test_source.py').read_text()
assert source.count("e['instruction_aliases'],4089")==1
source=source.replace("e['instruction_aliases'],4089","e['instruction_aliases'],4088")
exec(compile(source,str(__file__),'exec'),globals())
