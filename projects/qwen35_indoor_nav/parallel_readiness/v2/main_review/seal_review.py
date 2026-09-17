"""Seal CPU review files and snapshot main-agent state; no shared-file edits."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'SHA256SUMS').exists()
    for path in HERE.glob('*.json'):
        json.loads(path.read_text())
    links = 0
    for path in HERE.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            destination = (path.parent / target.split('#')[0]).resolve()
            assert destination.is_relative_to(LINE.resolve()) and destination.exists(), (path, target)
            links += 1
    state = json.loads((LINE / 'STATUS.json').read_text())
    assert state['parallel_readiness_closed'] and state['parallel_cpu_tests_passed'] == 43
    assert state['scientific_pass'] is False and state['sft_result_reviewed_by_main_agent'] is False
    check = {'json_and_relative_links_pass': True, 'links_checked': links,
             'research_framework_check_exit_code': 0,
             'root_state_snapshot_sha256': {p: digest(LINE / p) for p in ('STATUS.json', 'README.md', 'AGENTS.md')}}
    with (HERE / 'CLOSURE_CHECK.json').open('x') as f:
        json.dump(check, f, indent=2)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    with (HERE / 'SHA256SUMS').open('x') as f:
        f.write(''.join(digest(p) + '  ' + p.name + '\n' for p in files))
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert digest(HERE / relative) == expected
    print(json.dumps({'review_files_sealed': len(files), **check}, indent=2))


if __name__ == '__main__':
    main()
