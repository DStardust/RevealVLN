import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
path=HERE.parent/'short_revisit_v2/method.py'
spec=importlib.util.spec_from_file_location('short_revisit_v3_method',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ShortRevisitFactory=module.ShortRevisitFactory
