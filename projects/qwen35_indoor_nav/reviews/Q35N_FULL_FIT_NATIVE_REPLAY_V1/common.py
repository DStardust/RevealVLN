import importlib.util
import json
from pathlib import Path
_path = Path(__file__).with_name('source.py')
_spec = importlib.util.spec_from_file_location('native_source_common', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
exec(compile(_module.common(), str(Path(__file__)) + ':native-common', 'exec'), globals())


def read(path):
    return json.loads(Path(path).read_text())
