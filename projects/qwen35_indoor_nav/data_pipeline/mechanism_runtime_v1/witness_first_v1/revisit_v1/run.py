from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'assembly_v1'
def main():
    source=(BASE/'run.py').read_text()
    m=types.ModuleType('revisit_supervisor');m.__file__=str(BASE/'run.py')
    exec(compile(source,m.__file__,'exec'),m.__dict__);m.HERE=HERE;m.main()
if __name__=='__main__':main()
