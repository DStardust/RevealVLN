"""CPU-only batch_200…211 preparation. Main agent owns launch approval."""
import argparse
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('gpu1_prepare_adapter',Path(__file__).resolve().parent/'adapter.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
m=a.module('prepare.py')
prepare_one=m.prepare_one;prepare_many=m.prepare_many
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--indices',nargs='+',type=int);args=p.parse_args();prepare_many(args.indices)
