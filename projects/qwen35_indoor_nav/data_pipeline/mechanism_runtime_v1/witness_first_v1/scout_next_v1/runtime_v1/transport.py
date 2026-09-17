"""Exact, fail-closed source transport; no GPU work during imports or preparation."""
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent
SCOUT = NEXT.parent / 'scout_v1'
ROOT = next(p for p in HERE.parents if p.name == 'vla')
LINE = ROOT / 'projects/qwen35_indoor_nav'
RUNTIME = LINE / 'data_pipeline/mechanism_runtime_v1'
ENV = LINE / '.envs/q35n_habitat_v017_g0r'
GPUS = ((1, 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'),
        (2, 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'))


def shard_root(shard):
    assert type(shard) is int and shard in (0, 1), 'SHARD_ID'
    return NEXT / ('shard_' + str(shard))


def sha(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def exact(source, old, new):
    assert source.count(old) == 1, 'SOURCE_TRANSPORT_COUNT: ' + old
    return source.replace(old, new)


def verify_lock(path):
    values = json.loads(path.read_text())
    for name, expected in values.items():
        assert sha(ROOT / name) == expected, name


def approval_value(shard):
    gpu, _ = GPUS[shard]
    return dict(approved=True, shard=shard, gpu=gpu,
        prepared_input_lock_sha256=sha(NEXT / 'INPUT_LOCK.json'),
        runtime_input_lock_sha256=sha(HERE / 'INPUT_LOCK.json'))


def runtime_config(shard):
    folder = shard_root(shard)
    verify_lock(NEXT / 'INPUT_LOCK.json')
    verify_lock(HERE / 'INPUT_LOCK.json')
    approval = HERE / f'MAIN_AGENT_APPROVAL_SHARD_{shard}.json'
    assert json.loads(approval.read_text()) == approval_value(shard), 'EXACT_SHARD_APPROVAL'
    cfg = json.loads((folder / 'PREPARED_CONFIG.json').read_text())
    assert cfg['runtime_allowed'] is False and cfg['executable'] is False
    assert cfg['training_allowed'] is False and cfg['shard_id'] == shard
    assert (cfg['gpu_device'], cfg['gpu_uuid']) == GPUS[shard]
    assert ROOT / cfg['intended_output_root'] == folder / 'run_v1'
    cfg.update(runtime_allowed=True, executable=True, runtime_adapter_ready=True,
        main_agent_approval_sha256=sha(approval), runtime_source_lock_sha256=sha(HERE / 'INPUT_LOCK.json'))
    return cfg


def common_source(shard):
    source = (SCOUT / 'common.py').read_text()
    source = exact(source, 'HERE=Path(__file__).resolve().parent', f'HERE=Path({str(shard_root(shard))!r})')
    source = exact(source, 'RUNTIME=HERE.parents[1]', f'RUNTIME=Path({str(RUNTIME)!r})')
    # bank and budget_and_trace are siblings of the original scout, not NEXT/shard.
    source = exact(source, "str(HERE.parent/'budget_and_trace')", f'str(Path({str(SCOUT.parent / "budget_and_trace")!r}))')
    source = exact(source, "HERE.parent/'bank_cpu/bank.py'", f'Path({str(SCOUT.parent / "bank_cpu/bank.py")!r})')
    return source


def worker_source(shard):
    source = (SCOUT / 'worker.py').read_text()
    source = exact(source, 'HERE=Path(__file__).resolve().parent', f'HERE=Path({str(shard_root(shard))!r})')
    source = exact(source, 'from common import (', 'from scout_next_scoped_common import (')
    source = exact(source, "HabitatBackend(candidate['scene_glb'],1,candidate['roles']",
        f"HabitatBackend(candidate['scene_glb'],{GPUS[shard][0]},candidate['roles']")
    return source


def supervisor_source(shard):
    gpu, uuid = GPUS[shard]
    source = (RUNTIME / 'compact_loop_v2/run.py').read_text()
    edits = [
        ('HERE = Path(__file__).resolve().parent', f'HERE = Path({str(shard_root(shard))!r})'),
        ('LINE = HERE.parents[2]', f'LINE = Path({str(LINE)!r})'),
        ("UUID = 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'", f'UUID = {uuid!r}'),
        ("['nvidia-smi','-i','1','-q','-x']", f"['nvidia-smi','-i','{gpu}','-q','-x']"),
        ("assert sample['elapsed'] < 3000, 'WALL_BUDGET'", "assert sample['elapsed'] < 4500, 'WALL_BUDGET'"),
        ("sample['disk_bytes'] < 7*1024**3", "sample['disk_bytes'] < 8*1024**3"),
        ("snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)",
         "snapshot=gpu()\n                with (OUT/'GPU_SNAPSHOTS.jsonl').open('a') as stream: stream.write(json.dumps(snapshot)+'\\n')\n                upper=check_gpu(snapshot,proc.pid)"),
        ("str(HERE/'worker.py')", f"{str(HERE / 'worker.py')!r},'--shard','{shard}'"),
    ]
    for old, new in edits:
        source = exact(source, old, new)
    return source


def module_from_source(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    return module


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def transport_record(shard):
    return dict(shard=shard, gpu=GPUS[shard][0], gpu_uuid=GPUS[shard][1],
        source_strategy='exact_counted_string_substitutions_no_algorithm_change',
        here=str(shard_root(shard)), output_root=str(shard_root(shard) / 'run_v1'),
        environment=str(ENV), training_allowed=False,
        transformed_source_sha256={name: hashlib.sha256(fn(shard).encode()).hexdigest()
            for name, fn in [('common', common_source), ('worker', worker_source), ('supervisor', supervisor_source)]})
