from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
def main():
    path=HERE.parent/'assembly_v1/run.py'
    module=types.ModuleType('short_revisit_v3_supervisor');module.__file__=str(path)
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__);module.HERE=HERE;module.main()
if __name__=='__main__':main()
