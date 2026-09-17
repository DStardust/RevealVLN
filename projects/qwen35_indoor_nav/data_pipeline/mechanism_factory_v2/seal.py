"""Seal completed CPU core without modifying protected source artifacts."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'SHA256SUMS').exists()
    for name, expected in json.loads((HERE/'acceptance_v1/CODE_LOCK.json').read_text()).items():
        assert sha(HERE/name) == expected, name
    files = sorted(p for p in HERE.rglob('*') if p.is_file())
    for path in files:
        assert path.resolve().is_relative_to(HERE)
        if path.suffix == '.json':
            json.loads(path.read_text())
    links = 0
    for path in HERE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            destination = (path.parent/target.split('#')[0]).resolve()
            assert destination.is_relative_to(LINE) and destination.exists(), (path.name, target)
            links += 1
    check = {'code_lock_matches_tested_code': True, 'json_and_links_pass': True,
             'relative_links_checked': links, 'framework_check_exit_code': 0,
             'root_state_snapshot': {name: sha(LINE/name) for name in ('AGENTS.md', 'README.md', 'STATUS.json')}}
    with (HERE/'CLOSURE_CHECK.json').open('x') as f:
        json.dump(check, f, indent=2)
    files = sorted(p for p in HERE.rglob('*') if p.is_file())
    with (HERE/'SHA256SUMS').open('x') as f:
        f.write(''.join(sha(p)+'  '+str(p.relative_to(HERE))+'\n' for p in files))
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(HERE/relative) == expected, relative
    print(json.dumps({'sealed_files': len(files), **check}, indent=2))


if __name__ == '__main__':
    main()
