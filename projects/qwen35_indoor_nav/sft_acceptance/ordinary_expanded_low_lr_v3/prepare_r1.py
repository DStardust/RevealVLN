"""CPU preparation path correction; preserve the original pre-GPU attempt."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'prepare.py').read_text()
old="HERE.parent.parent.parent/'reviews/Q35N_ORDINARY_CONTINUE_V2/WORKFLOW_RESULT.json'"
assert raw.count(old)==1
raw=raw.replace(old,"HERE.parents[1]/'reviews/Q35N_ORDINARY_CONTINUE_V2/WORKFLOW_RESULT.json'")
anchor='changes={'
assert raw.count(anchor)==1
raw=raw.replace(anchor,'''changes={
 "save('CPU_TEST_RESULT.json',":"save('CPU_TEST_RESULT_R1.json',",''')
exec(compile(raw,str(HERE/'prepare_r1.py')+':v3-preparation','exec'),globals())
