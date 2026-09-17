"""Seal only this CPU planning directory; never alters shared nodes."""
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent
cpu = json.loads((out / 'CPU_RESULT.json').read_text())
tests = json.loads((out / 'TEST_RESULTS.json').read_text())
assert tests['passed'] and cpu['protected_files_unchanged']
result = dict(cpu, node='Q35N_MULTIFAMILY_CPU_PLAN_V2', decision='READY_FOR_MAIN_AGENT_REVIEW',
              tests_passed=tests['tests'], executable=False,
              unresolved=['generic compiler not implemented', 'runtime semantic eligibility and reachability unknown',
                          'action-count and unordered-history shortcuts need decisive cross-family audit',
                          'no heldout-house assignment', 'runtime authorization absent'],
              new_gpu_operations=0, new_simulator_replays=0, new_training_runs=0,
              sft_results_read=False, scope='metadata protocol and CPU dry-run only')
(out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
paths = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name for p in paths]
(out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
for line in lines:
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
print('sealed and verified', len(paths), 'files')
