"""Seal local deliverables and verify hashes; never execute simulator/model."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]

if __name__ == '__main__':
    target = OUT/'SHA256SUMS'
    assert not target.exists(), 'Do not rewrite a closed integrity manifest'
    result = json.loads((OUT/'result.json').read_text())
    assert result['runtime_pass'] and result['renderer_pass']
    assert result['scientific_pass'] is False and result['navigation_gain'] is None
    status = json.loads((LINE/'STATUS.json').read_text())
    assert status['g1_execution_approved'] is False
    assert not status['training_allowed'] and not status['method_experiments_allowed']
    for path in OUT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in OUT.rglob('*.json'):
        json.loads(path.read_text())
    listing = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            assert not path.is_symlink()
            listing.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUT)}\n')
    target.write_text(''.join(listing))
    check = subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT,capture_output=True,text=True)
    assert check.returncode == 0, check.stdout+check.stderr
    print(f'{len(listing)}/{len(listing)} delivery hashes verified; JSON, Python syntax and scientific guardrails passed.')
