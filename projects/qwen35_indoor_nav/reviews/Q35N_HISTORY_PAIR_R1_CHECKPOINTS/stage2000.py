"""Read-only numerical audit at 2000; no policy selection or training changes."""
import hashlib
from pathlib import Path
import re

SOURCE = Path(__file__).with_name('first200.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '2973bbab466fa80b472859ba34a8b3f834412b0f873758df992c599f72101e0c'
code = SOURCE.read_text()
for old, new in (
    ('stage200', 'stage2000'),
    ('checkpoint_000000200.pt', 'checkpoint_000002000.pt'),
    ('PASS_FIRST200_NUMERICAL_ONLY', 'PASS_STAGE2000_NUMERICAL_ONLY'),
    ('_FIRST200.json', '_STAGE2000.json'),
):
    assert code.count(old) == 1, old
    code = code.replace(old, new)
for old, new in (('200', '2000'), ('6400', '64000'), ('19200', '192000')):
    code, count = re.subn(r'\b' + old + r'\b', new, code)
    assert count > 0, old
exec(compile(code, str(SOURCE) + ':stage2000-counts-only', 'exec'), globals())
