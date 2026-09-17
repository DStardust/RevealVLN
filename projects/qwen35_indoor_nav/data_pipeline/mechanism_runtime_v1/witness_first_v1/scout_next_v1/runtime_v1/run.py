"""CPU import is inert. Each shard requires exact independent main-agent approval."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transport as t


def execute(shard):
    cfg = t.runtime_config(shard)
    out = t.shard_root(shard) / 'run_v1'
    out.mkdir(exist_ok=False)
    t.save(out / 'EXECUTION_CONFIG.json', cfg)
    t.save(out / 'TRANSPORT_APPLIED.json', t.transport_record(shard))
    supervisor = t.module_from_source('scout_next_scoped_supervisor', t.supervisor_source(shard), t.RUNTIME / 'compact_loop_v2/run.py')
    assert supervisor.HERE == t.shard_root(shard)
    assert supervisor.OUT == out and supervisor.ENV == t.ENV
    assert supervisor.UUID == t.GPUS[shard][1]
    supervisor.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    execute(parser.parse_args().shard)
