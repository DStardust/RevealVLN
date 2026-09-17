import importlib.util
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def main():
    source=(BASE/'run.py').read_text()
    assert source.count("sample['elapsed'] < 3000") == 1
    source=source.replace("sample['elapsed'] < 3000", "sample['elapsed'] < 2990")
    module=types.ModuleType('compact_v2_recovery_supervisor')
    module.__file__=str(BASE/'run.py')
    exec(compile(source,str(BASE/'run.py'),'exec'),module.__dict__)
    module.HERE=HERE
    module.OUT=HERE/'run_v1'
    module.main()

if __name__=='__main__':main()
