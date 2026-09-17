import runpy
from pathlib import Path
m=runpy.run_path(str(Path(__file__).with_name('accept.py')))
if __name__=='__main__':m['check'](200)
