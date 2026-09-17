from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transport
exec(compile(transport.original('safe_size.py'), __file__, 'exec'), globals())
