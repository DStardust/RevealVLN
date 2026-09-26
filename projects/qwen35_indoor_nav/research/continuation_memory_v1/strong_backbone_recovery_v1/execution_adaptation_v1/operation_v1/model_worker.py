"""Train one registered head, with the common source-bound trainer."""
import argparse
import fcntl
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]


def main(run, seed, mode, resume):
    import torch
    import common as u
    from data import load_rows
    from train import train_one
    torch.set_num_threads(4)
    config = u.read(run / 'TRAINING_CONFIG.json')
    folder = run / f'{mode}_s{seed}'
    folder.mkdir(exist_ok=True)
    with (folder / 'WORKER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {row['id']: row for row in load_rows(config['capture_run'], 'FIT')}
        if sorted(rows) != config['fit_ids']:
            raise ValueError('FIT_REGISTRATION_CHANGED')
        # One lossless device transfer per run; index metadata stays on CPU.
        device_keys = ('memory_features', 'actor_features', 'base_logits', 'executed_actions', 'known', 'targets')
        for row in rows.values():
            for key in device_keys:
                row[key] = row[key].to(config['device'])
        result = train_one(run, config, rows, seed, mode, resume=resume)
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--mode', choices=('CURRENT', 'DELTA'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.run.resolve(), args.seed, args.mode, args.resume)
