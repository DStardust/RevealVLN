"""One approved shard only; original discovery/trace/bank logic transported intact."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transport as t


def execute(shard):
    # Verify before loading any Habitat adapter or rendering anything.
    cfg = t.runtime_config(shard)
    module = t.module_from_source('scout_next_scoped_common', t.common_source(shard), t.SCOUT / 'common.py')
    module.runtime_config = lambda: cfg
    assert module.HERE == t.shard_root(shard) and module.ROOT == t.ROOT
    sys.modules['scout_next_scoped_common'] = module
    worker = t.module_from_source('scout_next_scoped_worker', t.worker_source(shard), t.SCOUT / 'worker.py')
    assert worker.HERE == t.shard_root(shard)
    worker.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    execute(parser.parse_args().shard)
