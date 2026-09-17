from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
exec(compile(transport.v2_source('merge_all.py'),str(__file__),'exec'),globals())
