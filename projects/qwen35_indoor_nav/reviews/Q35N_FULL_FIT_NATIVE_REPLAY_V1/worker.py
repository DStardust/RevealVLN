import importlib.util
from pathlib import Path
_path = Path(__file__).with_name('source.py')
_spec = importlib.util.spec_from_file_location('native_source_worker', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
exec(compile(_module.worker(), str(Path(__file__)) + ':native-model-prefix', 'exec'), globals())
