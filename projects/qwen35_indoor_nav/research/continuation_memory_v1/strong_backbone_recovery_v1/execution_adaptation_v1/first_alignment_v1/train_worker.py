"""Use the existing resumable trainer with the registered FIRST-only loss."""
import argparse
import fcntl
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from objective import legacy, objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--mode', choices=['CURRENT', 'DELTA'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    legacy.torch.set_num_threads(2)
    config = legacy.u.read(args.run / 'TRAINING_CONFIG.json')
    if config['version'] != 'FIRST_ALIGNED_TRAINING_V1' or config['action_scope'] != 'FIRST_ACTION_TOKENS_QUERY_START_MEMORY':
        raise ValueError('UNREGISTERED_FIRST_TRAINING')
    if config['device'].startswith('cuda'):
        props = legacy.torch.cuda.get_device_properties(0)
        legacy.torch.cuda.set_per_process_memory_fraction(config['gpu_memory_limit_gib'] * 2**30 / props.total_memory, 0)
    with (args.run / f'.{args.mode}_s{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {r['id']: r for r in legacy.load_rows(config['capture_run'], 'FIT')}
        if sorted(rows) != config['fit_ids']:
            raise ValueError('FIT_DATA_CHANGED')
        legacy.objective = objective
        legacy.train_one(args.run, config, rows, args.seed, args.mode, resume=args.resume)


if __name__ == '__main__':
    main()
