"""Audit paired action choice, exact pre-intervention equality and fixed gates."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run_001'
path = HERE.parent / 'ordinary_cycle_recovery_v1/review.py'
spec = importlib.util.spec_from_file_location('original_review', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    result = audit.read(RUN / 'RESULT.json')
    assert result['status'] == 'COMPLETE' and result['executed_episodes'] == 200
    traces = audit.traces(RUN)
    episodes = audit.read(HERE / 'EPISODES_PRIVILEGED.json')
    assert set(traces) == set(range(200))
    rows = [audit.audit(i, traces[i], traces[i + 100], episodes[i]['instruction']['instruction_text']) for i in range(100)]
    native, recovery = result['native'], result['recovery']
    delta = max(r['prefix_max_abs_logit_delta'] for r in rows)
    matched = all(r['prefix_mismatch'] is None for r in rows) and delta == 0
    paired = dict(sr=recovery['sr'] > native['sr'], spl=recovery['spl'] >= native['spl'],
                  ndtw=recovery['ndtw'] >= native['ndtw'] - .01)
    absolute = dict(sr=recovery['sr'] > .21, spl=recovery['spl'] >= .18016983923442284,
                    ndtw=recovery['ndtw'] >= .3558797007353029)
    review = dict(status='COMPLETE', role='Fixed engineering diagnostic, not a novelty or full validation claim',
        online_input_action_audit_passed=True, prefixes_and_logits_exact=matched, prefix_max_abs_logit_delta=delta,
        native=native, recovery=recovery, paired_gates=paired, original_absolute_gates=absolute,
        engineering_candidate_pass=matched and all(paired.values()) and all(absolute.values()),
        wins=[r['episode_id'] for r in rows if not r['baseline_success'] and r['recovery_success']],
        losses=[r['episode_id'] for r in rows if r['baseline_success'] and not r['recovery_success']],
        affected_episodes=sum(r['first_override'] is not None for r in rows),
        overrides=sum(r['overrides'] for r in rows), repeated_inputs=sum(r['repeated_inputs'] for r in rows),
        recovery_seconds=sum(p['cycle_seconds'] for i,t in traces.items() if i>=100 for p in t['policies']),episodes=rows)
    with (RUN / 'REVIEW.json').open('x') as stream:
        json.dump(review, stream, indent=2)
    print(json.dumps({k:v for k,v in review.items() if k not in ('episodes','native','recovery')}))


if __name__ == '__main__':
    main()
