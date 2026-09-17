"""Fresh output after streaming reader learned official trailing instruction_vocab."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
code=(HERE/'follower_manifest.py').read_text()
assert code.count("out=HERE/'follower_manifest_v1'")==1
code=code.replace("out=HERE/'follower_manifest_v1'","out=HERE/'follower_manifest_v2'")
exec(compile(code,str(__file__),'exec'),globals())
