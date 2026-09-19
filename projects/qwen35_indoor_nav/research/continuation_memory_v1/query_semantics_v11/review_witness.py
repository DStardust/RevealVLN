"""Read-only probe verification and explicitly post-hoc semantic coverage audit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    run = HERE/'witness_probe_run_001'
    result = read(run/'RESULT.json')
    rows = read(run/'EXAMPLES.json')['rows']
    data = read(HERE.parent/'multifamily_v7/DATA.json')
    tasks, fit_roles = {}, {'anchor_now': set(), 'terminal_now': set()}
    for family, audit in zip(data['families'], data['audit']['families']):
        config = read(LINE/audit['export']/'MANIFEST.json')['compiler_config']
        mapping = {}
        for task in config['tasks'].values():
            roles = {name: tuple(config['roles'][task[field]])
                     for name, field in (('anchor_now', 'anchor'), ('terminal_now', 'terminal'))}
            mapping[task['instruction']] = roles
            if family['split'] == 'fit':
                for name, role in roles.items():
                    fit_roles[name].add(role)
        tasks[family['family_id']] = mapping
    for row in rows:
        instruction = data['features'][row['feature']]['instruction']
        row['roles'] = tasks[row['family_id']][instruction]
    summaries = {}
    for key, value in result['results'].items():
        predictions = read(run/f'{key}_PREDICTIONS.json')
        assert len(predictions) == len(rows)
        coverage = {}
        for j, name in enumerate(('anchor_now', 'terminal_now')):
            for split in ('fit', 'check'):
                counts = dict(tp=0, tn=0, fp=0, fn=0)
                for row, logits in zip(rows, predictions):
                    if row['split'] != split or not row['mask'][j]:
                        continue
                    target, predicted = bool(row['y'][j]), logits[j] > 0
                    counts[('t' if target == predicted else 'f')+('p' if predicted else 'n')] += 1
                assert all(value['splits'][split][name][k] == v for k,v in counts.items())
            groups = {}
            for group in ('seen_target_role', 'unseen_target_role'):
                counts = dict(tp=0, tn=0, fp=0, fn=0)
                roles = set()
                for row, logits in zip(rows, predictions):
                    seen = row['roles'][name] in fit_roles[name]
                    if row['split']!='check' or not row['mask'][j] or seen != (group=='seen_target_role'):
                        continue
                    roles.add(row['roles'][name])
                    target, predicted = bool(row['y'][j]), logits[j] > 0
                    counts[('t' if target == predicted else 'f')+('p' if predicted else 'n')] += 1
                positives, negatives = counts['tp']+counts['fn'], counts['tn']+counts['fp']
                groups[group] = dict(**counts, roles=sorted(roles), positives=positives, negatives=negatives,
                    balanced_accuracy=(counts['tp']/positives+counts['tn']/negatives)/2 if positives and negatives else None)
            coverage[name] = groups
        summaries[key] = dict(check=value['splits']['check'], posthoc_coverage=coverage)
    means = {head: {name: sum(v['check'][name]['balanced_accuracy'] for k,v in summaries.items()
              if k.startswith(head+'_'))/3 for name in ('anchor_now','terminal_now')}
              for head in ('linear','mlp128')}
    report = dict(status='CPU_PROBE_VERIFIED_NO_METHOD_BENEFIT_CLAIM',
        result_sha256=hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest(),
        examples_sha256=hashlib.sha256((run/'EXAMPLES.json').read_bytes()).hexdigest(),
        all_six_prediction_files_recomputed=True, heads=means, results=summaries,
        fit_target_roles={name: sorted(roles) for name,roles in fit_roles.items()},
        role_coverage_analysis='Post-hoc explanatory grouping of the full unchanged CHECK denominator; not a new selected score or hypothesis test.',
        observations='Partial current-witness information is decodable; MLP nearly fits FIT but fails to transfer. Existing evidence does not isolate memory loss from semantic transfer difficulty.',
        next_decision='Do not expand memory or tune losses from these CHECK scores. A meaningful next method comparison needs a separately specified witness/semantic transfer control and independent physically certified families; retain all existing failures.',
        original_training_admission=False, research_claim_supported=False)
    with (HERE/'WITNESS_DIAGNOSIS.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(means))


if __name__ == '__main__':
    main()
