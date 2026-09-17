from pathlib import Path
import importlib.util
path=Path(__file__).resolve().parent/'runtime.py'
spec=importlib.util.spec_from_file_location('newhub_worker_transport',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':module.worker()
