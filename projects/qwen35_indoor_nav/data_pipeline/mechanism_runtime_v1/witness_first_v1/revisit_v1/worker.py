import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    method=load('witness_revisit_method',HERE/'method.py')
    worker=load('witness_revisit_worker',HERE.parent/'assembly_v1/worker.py')
    worker.HERE=HERE;worker.WitnessFactory=method.RevisitFactory
    worker.main()
if __name__=='__main__':main()
