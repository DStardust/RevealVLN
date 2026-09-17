import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    path=HERE.parent/'short_revisit_v2/worker.py'
    spec=importlib.util.spec_from_file_location('short_revisit_v3_worker',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE;module.main()
if __name__=='__main__':main()
