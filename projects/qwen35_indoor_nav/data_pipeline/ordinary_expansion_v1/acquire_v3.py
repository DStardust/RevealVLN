"""Use user-authorized proxyon environment; no credentials copied into source."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
code=(HERE/'acquire_v2.py').read_text()
assert code.count("'--proxy','http://127.0.0.1:27890',")==1
code=code.replace("'--proxy','http://127.0.0.1:27890',",'')
assert code.count("\"out=HERE/'acquisition_v2'\"")==1
code=code.replace("\"out=HERE/'acquisition_v2'\"","\"out=HERE/'acquisition_v3'\"")
exec(compile(code,str(__file__),'exec'),globals())
