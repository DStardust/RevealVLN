"""FIT-only event measures. Metadata selects supervision, never policy inputs."""
from collections import Counter, defaultdict
import random
import torch
from torch.nn import functional as F

ROLES = ('anchor', 'terminal')


class Pool:
    def __init__(self, records):
        self.records = list(records)
        self.cells = defaultdict(list)
        seen = set()
        for i, row in enumerate(self.records):
            key = (row['feature_index'], row['role'])
            if key in seen or row['split'] != 'FIT' or row['target'] not in (0, 1):
                raise ValueError('INVALID_OR_DUPLICATE_FIT_EVENT')
            seen.add(key)
            self.cells[(row['house'], row['role'])].append(i)
        self.keys = sorted(self.cells)
        if not self.keys:
            raise ValueError('EMPTY_EVENT_POOL')

    def mass(self):
        weights = [0.] * len(self.records)
        for indices in self.cells.values():
            for i in indices:
                weights[i] = 1 / (len(self.cells) * len(indices))
        return weights

    def sample(self, seed, step, count):
        rng = random.Random(f'event-v1:{seed}:{step}')
        return [rng.choice(self.cells[rng.choice(self.keys)]) for _ in range(count)]

    def sample_loss(self, net, cache, seed, step, count):
        rows = [self.records[i] for i in self.sample(seed, step, count)]
        return F.binary_cross_entropy_with_logits(predictions(net, cache, rows),
                    torch.tensor([r['target'] for r in rows], device=cache['features'].device, dtype=torch.float32))


def predictions(net, cache, records):
    device = cache['features'].device
    indices = torch.tensor([r['feature_index'] for r in records], device=device)
    roles = torch.tensor([ROLES.index(r['role']) for r in records], device=device)
    logits = net.events(net.core.feature_norm(cache['features'][indices]))
    return logits.gather(1, roles[:, None]).squeeze(1)


def original_mass(families, records, schedule):
    """Exact average of the original family-normalized class-weighted event BCE."""
    lookup = {(r['feature_index'], ROLES.index(r['role'])): i for i, r in enumerate(records)}
    counts = [[0, 0], [0, 0]]
    exposures = Counter(item['family'] for item in schedule)
    by_family = {}
    for family in families:
        if family['split'] != 'FIT':
            raise ValueError('NONFIT_WEIGHT_STATISTICS')
        rows = Counter()
        for sequence in family['sequences']:
            for feature, targets, masks in zip(sequence['features'], sequence['event_targets'], sequence['event_masks']):
                for role in range(2):
                    if not masks[role]:
                        continue
                    target = int(targets[role])
                    i = lookup[(feature, role)]
                    if records[i]['target'] != target or records[i]['house'] != family['house']:
                        raise ValueError('EVENT_LABEL_OR_HOUSE_MISMATCH')
                    counts[role][target] += masks[role]
                    rows[i] += masks[role]
        by_family[family['family_id']] = rows
    if any(min(count) <= 0 for count in counts):
        raise ValueError('MISSING_TRAIN_EVENT_CLASS')
    class_weights = [[sum(v) / (2*n) for n in v] for v in counts]
    included = sum(exposures[f] for f in by_family)
    if included <= 0:
        raise ValueError('EMPTY_FIT_SCHEDULE')
    result = [0.] * len(records)
    for family, rows in by_family.items():
        weighted = {i: n * class_weights[ROLES.index(records[i]['role'])][records[i]['target']] for i, n in rows.items()}
        denominator = sum(weighted.values())
        for i, value in weighted.items():
            result[i] += exposures[family] / included * value / denominator
    return result, dict(class_counts=counts, class_weights=class_weights,
                        family_exposures={f: exposures[f] for f in by_family}, schedule_steps=included)


def metrics(probabilities, records):
    if len(probabilities) != len(records) or any(not 0 <= float(p) <= 1 for p in probabilities):
        raise ValueError('INVALID_PROBABILITY_OR_METRIC_DENOMINATOR')
    result = {}
    for role in ROLES:
        values = [(float(p), r['target']) for p, r in zip(probabilities, records) if r['role'] == role]
        positive = sum(y for p, y in values); negative = len(values) - positive
        if not positive or not negative:
            raise ValueError('UNIDENTIFIABLE_ROLE_METRIC')
        tp = sum(p >= .5 and y == 1 for p, y in values)
        fp = sum(p >= .5 and y == 0 for p, y in values)
        result[role] = dict(n=len(values), positive=positive, negative=negative, tp=tp, fp=fp,
                            recall=tp / positive, fpr=fp / negative,
                            brier=sum((p-y)**2 for p, y in values) / len(values))
    return result
