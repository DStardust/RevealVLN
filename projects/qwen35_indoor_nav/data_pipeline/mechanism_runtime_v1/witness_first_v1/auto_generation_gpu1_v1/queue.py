"""Independent CPU queue for GPU1: reserves all frozen GPU2 candidates."""
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('gpu1_queue_adapter',Path(__file__).resolve().parent/'adapter.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
m=a.module('queue.py')
freeze=m.freeze;choose=m.choose;normalize=m.normalize;semantic=m.semantic;same_program=m.same_program
if __name__=='__main__':freeze()
