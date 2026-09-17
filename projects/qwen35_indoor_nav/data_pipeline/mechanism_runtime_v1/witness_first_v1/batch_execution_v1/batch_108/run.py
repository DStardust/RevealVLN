import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('auto_generation_frozen_entry','/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/auto_generation_v1/transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
if __name__=='__main__':t.run_main(HERE)
