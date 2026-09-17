"""CPU preparation entrypoint. No automatic simulator/model execution."""
import argparse
import hashlib
import json
from pathlib import Path

from planning import select_candidates, DEFAULT_LIMITS

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SOURCE = LINE/'parallel_readiness/v2/multifamily_plan'


def verify_seal(directory):
    count = 0
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split(None, 1)
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('SEALED_INPUT_HASH_MISMATCH:'+relative)
        count += 1
    return count


def prepare(output):
    output = Path(output).resolve()
    if output == HERE or not output.is_relative_to(HERE):
        raise ValueError('OUTPUT_SCOPE')
    verified = verify_seal(SOURCE)
    records = json.loads((SOURCE/'CANDIDATES.json').read_text())
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    selected = select_candidates(records)
    roles = protocol['task_instance']['stage_P0']
    # Deterministic wording from registered semantic tuples, not free-form relabeling.
    nouns = {'tv_monitor': '电视', 'sink': '水槽', 'bed': '床', 'chair': '椅子'}
    rooms = {'living room': '客厅', 'kitchen': '厨房', 'bedroom': '卧室', 'dining room': '餐厅'}
    tasks = {}
    for key in ('anchor_A', 'anchor_B'):
        category, room = roles[key]
        end_category, end_room = roles['terminal']
        tasks['task_'+key[-1]] = {'anchor': key, 'terminal': 'terminal',
            'instruction': f'先连续两帧看见{rooms[room]}内的{nouns[category]}，再连续两帧看见{rooms[end_room]}内的{nouns[end_category]}，然后立即停止。'}
    result = {'node': 'GENERIC_FACTORY_CPU_PREPARATION_V2', 'executable': False,
              'candidate_count': len(selected), 'candidates': selected, 'roles': roles,
              'tasks': tasks, 'limits': DEFAULT_LIMITS,
              'eligible': None, 'runtime_u_positions': None, 'gpu_device': None,
              'source_files_verified': verified,
              'source_protocol_sha256': hashlib.sha256((SOURCE/'protocol.json').read_bytes()).hexdigest(),
              'pending': ['scene asset hash + semantic mapping', 'route-coordinate resolution + legal start validation',
                          'Habitat backend adapter + resource watchdog', 'explicit runtime authorization and GPU lease',
                          'certification export/schema-loader v4 integration'],
              'new_physical_families': 0, 'scientific_pass': False}
    output.mkdir(parents=True, exist_ok=False)
    with (output/'PREPARED_P0.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'certify'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command != 'prepare':
        parser.error('RUNTIME_NOT_ADMITTED: only CPU prepare implemented in this node; no simulator import performed')
    if args.output is None:
        parser.error('--output is required')
    result = prepare(args.output)
    print(json.dumps({'candidate_count': result['candidate_count'], 'executable': False,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
