"""One bounded GPU parity check of the standalone entry against recorded logits."""
import json
import math
from pathlib import Path
import runpy
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]


def main():
    helpers = runpy.run_path(str(LINE / 'closed_loop_bench/ordinary_cycle_recovery_v1/launch.py'))
    plan = json.loads((HERE / 'ACCEPTANCE_PLAN.json').read_text())
    before = helpers['gpu_snapshot']()[plan['gpu']]
    assert before['uuid'] == plan['gpu_uuid'] and not before['contexts'] and before['memory_mib'] < 128
    output = HERE / 'acceptance_predictions.jsonl'
    began = time.monotonic()
    peak_gpu = peak_rss = 0
    failure = None
    with (HERE / 'acceptance_worker.log').open('x') as log:
        proc = subprocess.Popen([str(LINE / '.envs/q35n_qwen_g2_v1/bin/python3'), '-I', '-B',
            str(HERE / 'predict.py'), '--gpu', str(plan['gpu']), '--input', str(HERE / 'ACCEPTANCE_INPUTS.jsonl'),
            '--output', str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = helpers['identity'](proc.pid)
        try:
            while proc.poll() is None:
                members = helpers['group_members'](proc.pid)
                pids = {p['pid'] for p in members}
                gpu = helpers['gpu_snapshot']()[plan['gpu']]
                peak_gpu = max(peak_gpu, gpu['memory_mib'])
                peak_rss = max(peak_rss, sum(p['rss_bytes'] for p in members))
                if any(p['pid'] not in pids for p in gpu['contexts']):
                    failure = 'Foreign context appeared'
                elif time.monotonic() - began > plan['wall_seconds']:
                    failure = 'Wall budget'
                elif peak_gpu > plan['max_gpu_gib'] * 1024 or peak_rss > plan['max_rss_gib'] * 1024**3:
                    failure = 'Memory budget'
                if failure:
                    break
                time.sleep(2)
        finally:
            cleanup = helpers['terminate_owned'](proc, identity)
    after = helpers['gpu_snapshot']()[plan['gpu']]
    receipt = dict(returncode=proc.returncode, failure=failure, wall_seconds=time.monotonic() - began,
                   peak_gpu_mib=peak_gpu, peak_rss_bytes=peak_rss, cleanup=cleanup, gpu_after=after,
                   optimizer_updates=0, source_sha256=plan['source_sha256'])
    passed = failure is None and proc.returncode == 0
    if passed:
        expected = json.loads(Path(plan['expected']).read_text())
        expected = [row for check in expected['checks'] for row in check['single_logits']]
        actual = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(actual) == len(expected) == 16
        delta = [a - b for row, ref in zip(actual, expected) for a, b in zip(row['logits'], ref)]
        max_abs = max(map(abs, delta))
        relative = math.sqrt(sum(x*x for x in delta) / sum(x*x for row in expected for x in row))
        actions_match = all(max(range(4), key=lambda i: row['logits'][i]) == max(range(4), key=lambda i: ref[i])
                            for row, ref in zip(actual, expected))
        passed = actions_match and max_abs <= .15 and relative <= .03
        receipt.update(inputs=16, actions_match=actions_match, max_abs_logit_delta=max_abs,
                       relative_l2=relative, first_request_seconds=actual[0]['preprocessing_and_forward_seconds'],
                       subsequent_request_mean_seconds=sum(r['preprocessing_and_forward_seconds'] for r in actual[1:]) / 15)
    receipt['passed'] = passed
    with (HERE / 'DEPLOYMENT_ACCEPTANCE.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
