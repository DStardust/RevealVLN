from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import recover
source=recover.source_file('prepare.py')
source=recover.exact(source,"paths=list(HERE.glob('*.py'))",
    "paths=list(HERE.glob('*.py'))+[recover.SOURCE/'INPUT_LOCK.json',recover.SOURCE/'recover.py',recover.SOURCE/'prepare.py']")
exec(compile(source,str(__file__),'exec'),globals())
