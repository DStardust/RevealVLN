"""Bootstrap, repair, resume, and publish the V16 TEST-family continuation.

The old formal run is treated as immutable evidence.  This script creates a
new run containing the eight complete house records, performs only the
versioned rqfALeAoiTq repair collector, then invokes the unchanged V16
pipeline.  It is intended to run as the child of ``standalone.py`` under
systemd; every expensive stage has the normal V16 checkpoint/resume path.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v16_common import *
import pipeline


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PYTHON = ROOT / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
SIMPY = LINE / '.envs/q35n_habitat_v017_g0r/bin/python3'
OLD_RUN_ID = 'v16_formal_001'
OLD_UNIT = 'q35n-v16-formal-20260920-01.service'
REPAIR_VERSION = 'v16_rqf_full_neutral_yaw_search_v1'
HOUSE = 'rqfALeAoiTq'


def _copy_bytes(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != blob:
            raise ValueError('SOURCE_SNAPSHOT_MISMATCH:' + str(destination))
        return
    temporary = destination.with_name(destination.name + '.tmp.' + str(os.getpid()))
    temporary.write_bytes(blob)
    os.replace(temporary, destination)


def _assert_old_run(source):
    if subprocess.run(['systemctl', 'is-active', '--quiet', OLD_UNIT]).returncode == 0:
        raise RuntimeError('OLD_V16_SERVICE_STILL_ACTIVE')
    status = read(source / 'STATUS.json')
    if status.get('status') != 'STOPPED':
        raise ValueError('OLD_RUN_STATUS_NOT_STOPPED')
    records = [read(path) for path in sorted(source.glob('HOUSE_*.json'))]
    by_split = {}
    for record in records:
        house = next(h for h in read(HERE / 'DATA_MANIFEST.json')['houses'] if h['house'] == record['house'])
        by_split.setdefault(house['split'], []).append(record)
    counts = {split: sum(len(record['families']) for record in rows) for split, rows in by_split.items()}
    if counts.get('FIT') != 16 or counts.get('DEV') != 2 or counts.get('TEST') != 6:
        raise ValueError('OLD_RUN_COVERAGE_CHANGED:' + repr(counts))
    failed = next(record for record in records if record['house'] == HOUSE)
    if failed.get('complete') or failed.get('families'):
        raise ValueError('OLD_RQF_RECORD_NOT_SHORTFALL')
    if len(json_records(source / ('PROPOSALS_' + HOUSE + '.json'))) != 117:
        raise ValueError('OLD_RQF_CANDIDATE_COUNT_CHANGED')
    if (source / 'train').exists() or (source / 'evaluate').exists():
        raise ValueError('OLD_RUN_ALREADY_ENTERED_MODEL_STAGE')


def json_records(path):
    return read(path)


def _gpu_processes(config):
    query = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    )
    rows = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3 or fields[0] != config['gpu_uuid']:
            continue
        pid = int(fields[1])
        command = ''
        try:
            command = Path('/proc').joinpath(str(pid), 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        except OSError:
            command = 'PROC_GONE'
        rows.append(dict(uuid=fields[0], pid=pid, used_memory_mib=int(fields[2]), command=command))
    return rows


def _gpu_inventory(config):
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    )
    inventory = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 5:
            continue
        uuid = fields[1]
        local = dict(index=int(fields[0]), uuid=uuid, total_mib=int(fields[2]),
                     used_mib=int(fields[3]), free_mib=int(fields[4]))
        local['processes'] = _gpu_processes(dict(config, gpu_uuid=uuid))
        local['idle'] = not local['processes']
        inventory.append(local)
    return inventory


def wait_for_idle_gpu(run, baseline):
    """Select an actually idle card; never stop or contend with another process."""
    marker = run / 'GPU_WAIT_ALL.jsonl'
    required_mib = int(baseline['model_memory_gib'] * 1024)
    while True:
        inventory = _gpu_inventory(baseline)
        candidates = [row for row in inventory if row['idle'] and row['free_mib'] >= required_mib]
        snapshot = dict(unix=time.time(), required_free_mib=required_mib, inventory=inventory,
                        candidates=[dict(index=row['index'], uuid=row['uuid'], free_mib=row['free_mib'])
                                    for row in candidates],
                        decision='PROCEED' if candidates else 'WAIT_EXTERNAL_PROCESS')
        append(marker, snapshot)
        if candidates:
            selected = sorted(candidates, key=lambda row: (row['index'], row['uuid']))[0]
            config = dict(baseline, gpu=selected['index'], gpu_uuid=selected['uuid'])
            immutable(run / 'BASE_PROTOCOL.json', baseline)
            write(run / 'PROTOCOL.json', config)
            immutable(run / 'GPU_ASSIGNMENT.json', dict(
                source_protocol_sha256=sha(run / 'BASE_PROTOCOL.json'), selected=selected,
                no_external_process_at_selection=True, selection_unix=time.time(),
            ))
            return config
        time.sleep(30)


def wait_for_selected_gpu(run, config):
    """After binding, wait if an external process appears; do not rebind mid-run."""
    marker = run / 'GPU_WAIT_SELECTED.jsonl'
    while True:
        rows = _gpu_processes(config)
        append(marker, dict(unix=time.time(), gpu_uuid=config['gpu_uuid'], processes=rows,
                            decision='PROCEED' if not rows else 'WAIT_EXTERNAL_PROCESS'))
        if not rows:
            return
        time.sleep(30)


def bootstrap(run, source):
    _assert_old_run(source)
    baseline = read(source / 'PROTOCOL.json')
    run.mkdir(parents=True, exist_ok=True)
    if not (run / 'BASE_PROTOCOL.json').exists():
        immutable(run / 'BASE_PROTOCOL.json', baseline)
    if (run / 'GPU_ASSIGNMENT.json').exists():
        config = read(run / 'PROTOCOL.json')
        assignment = read(run / 'GPU_ASSIGNMENT.json')['selected']
        if config['gpu_uuid'] != assignment['uuid'] or config['gpu'] != assignment['index']:
            raise ValueError('GPU_ASSIGNMENT_PROTOCOL_MISMATCH')
    else:
        if (run / 'PROTOCOL.json').exists() and read(run / 'PROTOCOL.json') != baseline:
            raise ValueError('BASE_PROTOCOL_CHANGED_BEFORE_GPU_SELECTION')
        config = wait_for_idle_gpu(run, baseline)
    for record_path in sorted(source.glob('HOUSE_*.json')):
        record = read(record_path)
        if record['house'] == HOUSE:
            immutable(run / ('PRIOR_HOUSE_' + HOUSE + '.json'), record)
        elif record.get('complete'):
            immutable(run / record_path.name, record)
    lock = source_lock()
    immutable(run / 'SOURCE_LOCK.json', lock)
    for relative, expected in lock['files'].items():
        source_path = LINE / relative
        if sha(source_path) != expected:
            raise ValueError('SOURCE_CHANGED_DURING_BOOTSTRAP:' + relative)
        _copy_bytes(source_path, run / 'source' / relative)
    source_records = {path.name: sha(path) for path in sorted(source.glob('HOUSE_*.json'))}
    immutable(run / 'REUSED_PHYSICAL_ASSETS.json', dict(
        source=str(source), source_protocol_sha256=sha(source / 'PROTOCOL.json'),
        source_house_records=source_records, source_status_sha256=sha(source / 'STATUS.json'),
        policy_models_used_for_selection=False, completed_families_reused=24,
        incomplete_house=HOUSE, old_failures_read_only=True,
        treatment_comparison='All arms share all selected actual trajectories; repair changes only physical candidate discovery.',
    ))
    immutable(run / 'REPAIR_SPEC.json', dict(
        version=REPAIR_VERSION, source_run=str(source), source_candidate_count=117,
        target_families=2, accepted_before_repair=0,
        changed_path='try every frozen neutral_start_yaw_candidates value before rejecting a proposal',
        extra_hubs=24, repair_max_proposals=256,
        unchanged_requirements={
            'SEE2': 'unchanged evaluator threshold and two consecutive observations',
            'collision': 'any collision rejects the physical attempt',
            'active_stop': 'unchanged actual S action and terminal witness requirement',
            'decision_budget': 500, 'history_cap': 240,
            'information_isolation': 'other anchor event rejects the history',
            'model_scores_read': False,
        },
        proposal_order='old failed proposals in source order, then deterministic pathfinder continuation; no score ordering',
        completed_reuse_policy='all eight complete HOUSE records copied by value; no completed family recollection',
    ))
    write(run / 'BUDGET_FREEZE.json', dict(
        old_run_gpu_session_hours=read(source / 'STATUS.json').get('gpu_session_hours'),
        repair_protocol_gpu_session_hours=config['gpu_session_hours'],
        max_continuous_hours=config['max_session_hours'], gpu_uuid=config['gpu_uuid'],
        gpu_ordinal=config['gpu'], model_stage_not_started=True,
    ))
    pipeline.preflight(run, config)
    write(run / 'PREFLIGHT_COMPLETE.json', dict(phase='preflight', runtime_gpu=config['gpu_uuid'],
                                                completed_unix=time.time()), True)
    report = run / 'REPAIR_REPORT_ZH.md'
    if not report.exists():
        report.write_text(
            '# V16 rqfALeAoiTq 修复运行\n\n'
            f'- 来源旧 run：`{source}`（只读）\n'
            f'- 修复版本：`{REPAIR_VERSION}`\n'
            '- 已复用 24 个已认证族；只重新搜索缺失 TEST 屋的 2 个族。\n'
            '- 不读取模型分数，不改变 SEE2、碰撞、主动 STOP、500 决策或信息隔离门槛。\n'
            '- 修复方式：穷举冻结中性朝向；仍不足时按确定性 pathfinder 流追加 hub。\n',
            encoding='utf-8',
        )
    return config


def run_repair(run, config):
    if not (run / 'REPAIR_COLLECTION_COMPLETE.json').exists():
        wait_for_selected_gpu(run, config)
        pipeline.execute(
            run, config, 'collect_repair_test',
            [HERE / 'collect_repair_v1.py', run, 'TEST'], SIMPY, gpu=True,
        )
    if not (run / 'REPAIR_COLLECTION_COMPLETE.json').exists():
        raise RuntimeError('REPAIR_COLLECTION_NOT_COMPLETE')


def run_full_pipeline(run, run_id):
    marker = run / 'PIPELINE_RESUME_COMPLETE.json'
    if marker.exists():
        return
    command = [str(PYTHON), '-I', '-S', '-B', str(HERE / 'pipeline.py'),
               '--config', str(run / 'PROTOCOL.json'), '--run-id', run_id,
               '--phase', 'all', '--resume']
    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise SystemExit(result.returncode)
    write(marker, dict(run=str(run), completed_unix=time.time(), method_scores_read=False), True)


def publish(run):
    if (run / 'REPAIR_PUBLISHED.json').exists():
        return
    command = [str(PYTHON), '-I', '-S', '-B', str(HERE / 'publish.py'),
               '--message', 'Repair V16 rqfALeAoiTq physical collection and resume formal pipeline']
    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise SystemExit(result.returncode)
    write(run / 'REPAIR_PUBLISHED.json', dict(completed_unix=time.time(), branch='codex/q35n-grounded-state-v16-20260920'), True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, default=HERE / 'runs' / OLD_RUN_ID)
    parser.add_argument('--run-id', default='v16_repair_rqf_001')
    args = parser.parse_args()
    source = args.source_run.resolve(); run = HERE / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'REPAIR.lock').open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        config = bootstrap(run, source)
        run_repair(run, config)
        wait_for_selected_gpu(run, config)
        run_full_pipeline(run, args.run_id)
        publish(run)
        status = read(run / 'STATUS.json') if (run / 'STATUS.json').exists() else {}
        status.update(status='REPAIR_AND_RESUME_PUBLISHED', run=str(run),
                      repair_version=REPAIR_VERSION, unix=time.time())
        write(run / 'STATUS.json', status, True)


if __name__ == '__main__':
    main()
