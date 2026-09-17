import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch_05r1_readiness',HERE.parent/'readiness_v1.py')
ready=importlib.util.module_from_spec(spec);spec.loader.exec_module(ready)
if __name__=='__main__':ready.run_main(HERE)
