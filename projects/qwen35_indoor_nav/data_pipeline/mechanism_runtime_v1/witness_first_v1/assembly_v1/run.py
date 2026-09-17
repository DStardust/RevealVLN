from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
def main():
    source=(RUNTIME/'compact_loop_v2/run.py').read_text()
    assert source.count("'nvidia-smi','-i','1'")==1
    source=source.replace("'nvidia-smi','-i','1'","'nvidia-smi','-i','2'")
    m=types.ModuleType('witness_gpu2_supervisor');m.__file__=str(RUNTIME/'compact_loop_v2/run.py')
    exec(compile(source,m.__file__,'exec'),m.__dict__)
    m.HERE=HERE;m.OUT=HERE/'run_v1';m.LINE=RUNTIME.parents[1];m.ENV=m.LINE/'.envs/q35n_habitat_v017_g0r'
    m.UUID='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
    m.main()
if __name__=='__main__':main()
