import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch_05_shared',HERE.parent/'shared.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
if __name__=='__main__':shared.worker_main(HERE)
