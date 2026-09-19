"""Recompute every saved witness confusion table without loading a model."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    run = HERE/'screen_cpu_run_001'
    result = read(run/'RESULT.json')
    rows = read(HERE/'SCREEN.json')['rows']
    partitions = sorted({row['partition'] for row in rows})
    verified = {}
    for key, measured in result['results'].items():
        predictions = read(run/f'{key}_PREDICTIONS.json')
        assert len(predictions) == len(rows)
        assert all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in predictions)
        for field, stored_field in [('partition', 'partitions'), ('house', 'houses'), ('family_id', 'families')]:
            for value in sorted({row[field] for row in rows}):
                for j, target in enumerate(('anchor_now', 'terminal_now')):
                    counts = dict(tp=0, tn=0, fp=0, fn=0)
                    for row, logits in zip(rows, predictions):
                        if row[field] != value or not row['mask'][j]:
                            continue
                        truth, predicted = bool(row['y'][j]), logits[j] > 0
                        counts[('t' if truth == predicted else 'f')+('p' if predicted else 'n')] += 1
                    stored = measured[stored_field][value][target]
                    assert all(stored[k] == v for k, v in counts.items())
                    positive, negative = counts['tp']+counts['fn'], counts['tn']+counts['fp']
                    balanced = (counts['tp']/positive+counts['tn']/negative)/2 if positive and negative else None
                    assert stored['balanced_accuracy'] == balanced
        verified[key] = measured['partitions']
    means = {head: {partition: {target: sum(v[partition][target]['balanced_accuracy']
             for k, v in verified.items() if k.startswith(head+'_'))/3
             for target in ('anchor_now', 'terminal_now')} for partition in partitions}
             for head in ('linear', 'mlp128')}
    receipt = read(HERE/'features_run_001/FEATURE_RESULT.json')
    launch = read(HERE/'features_run_001/LAUNCH_RESULT.json')
    assert launch['status'] == 'COMPLETE' and receipt['parameters_unchanged']
    report = dict(status='SCREEN_COMPLETE_NO_POLICY_BENEFIT_MEASURED', means=means,
        all_prediction_partition_house_family_counts_recomputed=True, runs=verified,
        result_sha256=hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest(),
        real_qwen_forwards=receipt['real_qwen_forwards'], reused_features=receipt['reused_rows'],
        gpu_hours=launch['gpu_seconds']/3600, cpu_wall_seconds=result['seconds'],
        diagnostic_optimizer_updates=result['diagnostic_head_updates'], runtime_policy_updates=0,
        original_training_admission=False, publication_ready=False,
        interpretation='Expanded FIT pool does not yield a stable transferable anchor witness probe in this finite screen. This does not prove the representation lacks the information. Terminal signal is stronger. No full-pool training or method-benefit claim follows.',
        uncertainty='Only three additional houses, correlated histories/frames and fixed sampled diagnostic rows. Not blind base-model testing or independent per-frame trials.')
    with (HERE/'SCREEN_REVIEW.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(means))


if __name__ == '__main__':
    main()
