import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('bulk_scout_runtime',Path(__file__).with_name('runtime.py'))
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=[0],required=True)
    runtime.worker(parser.parse_args().shard)
