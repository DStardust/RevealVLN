from pathlib import Path
HERE=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(HERE))
import transport
source=transport.v2_source('test_safety_inherited.py')
exec(compile(source,str(__file__),'exec'),globals())
