"""Read-only closure/producer observation; outputs only immutable monitor snapshots."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if p.name == 'vla')
DATA = ROOT / 'projects/qwen35_indoor_nav/data_pipeline'
SCOUT = HERE.parent / 'run_v1'
ORDINARY = DATA / 'ordinary_scale_v1'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordinary():
    jobs = read(ORDINARY / 'JOBS.json')
    ledger = rows(ORDINARY / 'LEDGER.jsonl')
    job_ids = {x['job_id'] for x in jobs}
    assert len(ledger) == len(job_ids) == 1000
    assert len({x['job_id'] for x in ledger}) == 1000
    assert {x['job_id'] for x in ledger} == job_ids
    accepted, quarantined, indices, audits, inputs = set(), set(), [], [], {}
    for ap in sorted((ORDINARY / 'shards').glob('*.audit.json')):
        a = read(ap)
        ip = ap.with_name(ap.name.replace('.audit.json', '.jsonl'))
        assert sha(ip) == a['index_sha256'] and a['integrity_pass']
        rr = rows(ip)
        route_ids = {x['job_id'] for x in rr}
        assert not accepted.intersection(route_ids)
        assert len(route_ids) == a['certified_routes']
        assert len(rr) == a['instruction_records']
        assert sum(x['decisions'] for x in rr) == a['instruction_conditioned_decisions']
        qp = ORDINARY / a['quarantine_manifest']
        qq = read(qp)
        assert len(qq) == a['quarantined_routes']
        assert all(x['status'] == 'QUARANTINED_NOT_TRAINING_DATA' for x in qq)
        accepted.update(route_ids)
        quarantined.update(x['job_id'] for x in qq)
        indices.extend(rr)
        audits.append(a)
        for p in [ap, ip, qp]:
            inputs[str(p.relative_to(ROOT))] = sha(p)
    assert accepted.isdisjoint(quarantined)
    replay = {x['job_id'] for x in ledger if x['status'] == 'CERTIFIED'}
    assert accepted | quarantined == replay
    assert len({x['policy_file'] for x in indices}) == len(indices)
    progress = read(ORDINARY / 'recovery_v4/PROGRESS.json')
    assert len(accepted) == progress['audited_routes']
    assert len(indices) == progress['audited_instruction_records']
    result = read(ORDINARY / 'recovery_v4/run/RESULT.json')
    restore = read(ORDINARY / 'recovery_v4/run/RESTORATION.json')
    pid = restore['pid']
    cmd = Path('/proc', str(pid), 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    cwd = str(Path('/proc', str(pid), 'cwd').resolve())
    pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t',
        'vla_idle_occupancy_20260904:2.0', '#{pane_id} #{pane_pid}'], text=True).strip()
    assert cmd == restore['command'] and cwd == str(ROOT)
    assert pane == '%150 ' + str(pid)
    tree = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    gpu = next(g for g in tree.findall('gpu') if g.findtext('uuid') == 'GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b')
    processes = [{c.tag: c.text for c in p} for p in gpu.findall('./processes/process_info')]
    assert any(x['pid'] == str(pid) for x in processes)
    return {'ledger_terminal_routes': len(ledger), 'replay_certified': len(replay),
        'strict_routes': len(accepted), 'quarantined_routes': len(quarantined),
        'strict_instruction_records': len(indices),
        'strict_conditioned_decisions': sum(x['decisions'] for x in indices),
        'strict_by_source': dict(collections.Counter(x['source'] for x in indices)),
        'shards': len(audits), 'supervisor': result, 'restoration_independently_verified': True,
        'holder': {'pid': pid, 'command': cmd, 'cwd': cwd, 'pane': pane, 'gpu_processes': processes},
        'input_sha256': inputs}


def scout():
    houses = []
    for h in sorted((SCOUT / 'houses').iterdir()):
        if not h.is_dir():
            continue
        rec = {'house': h.name, 'status': 'ACTIVE_OR_NOT_CLOSED', 'program_sets': []}
        rp = h / 'result.json'
        if rp.exists():
            rec['result'] = read(rp)
            rec['status'] = rec['result']['status']
        for p in sorted(h.glob('*PROGRAMS.json')):
            j = read(p)
            proposals = j['proposals']
            rec['program_sets'].append({'file': str(p.relative_to(ROOT)), 'sha256': sha(p),
                'input_traces': j['input_traces'], 'compatible_anchor_pairs': j['compatible_anchor_pairs'],
                'enumerated': j['enumerated_programs'], 'returned': len(proposals),
                'all_components_stored': sum(x['all_components_stored'] for x in proposals),
                'irrelevant_with_2F': sum(x['irrelevant_has_at_least_2F'] for x in proposals),
                'no_claim_of_physical_certification': all(not x['physical_replay_certified'] for x in proposals)})
        houses.append(rec)
    out = {'progress_snapshot': read(SCOUT / 'PROGRESS.json'), 'houses': houses,
        'supervisor_closed': (SCOUT / 'SUPERVISOR_RESULT.json').exists(),
        'resource_snapshot': rows(SCOUT / 'RESOURCE_SAMPLES.jsonl')[-1],
        'scope': 'Live snapshot unless explicitly closed; candidate programs are not generated families.'}
    for name in ['result.json', 'SUPERVISOR_RESULT.json']:
        if (SCOUT / name).exists():
            out[name] = read(SCOUT / name)
    return out


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    report = {'timestamp_utc': now.isoformat(), 'ordinary': ordinary(), 'scout': scout(),
        'scientific_pass': False, 'training_started_by_monitor': False,
        'external_process_mutations': 0, 'live_source_mutations': 0}
    dst = HERE / ('snapshot_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    with dst.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(dst)
    print(json.dumps({k: v for k, v in report['ordinary'].items() if k not in {'input_sha256', 'holder'}}, ensure_ascii=False))
    print(json.dumps(report['scout'], ensure_ascii=False))


if __name__ == '__main__':
    main()
