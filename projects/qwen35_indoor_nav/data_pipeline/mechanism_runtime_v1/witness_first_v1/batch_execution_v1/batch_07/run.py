import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('batch07_clock_transport',HERE.parent/'budget_batched_transport_v1/transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
if __name__=='__main__':t.run_main(HERE)
