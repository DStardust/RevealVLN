"""Seal only local loader delivery; verify accepted source again without edits."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
import loader

before = json.loads((root / 'PROTECTED_BEFORE.json').read_text())
assert loader.verify_protected() == before
for name, expected in json.loads((root / 'CODE_LOCK.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
entries = []
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != 'SHA256SUMS':
        entries.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name)
(root / 'SHA256SUMS').write_text('\n'.join(entries) + '\n')
for entry in entries:
    expected, name = entry.split('  ', 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
print(json.dumps({'delivery_files_verified': len(entries), 'protected': before}))
