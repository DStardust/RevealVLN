"""Dual-kernel training-equivalence comparison: fallback vs fla, same data order.

Gates (frozen in ACCEPTANCE.json acceptance_v2 before any run):
  - per-update relative CE divergence <= ce_band after warmup
  - final-window mean CE relative difference <= ce_final
  - per-class recall difference on the final window <= recall_band
FAIL stays FAIL; no threshold is edited after observation.
"""
import argparse
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_series(path):
    events = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    series = {}
    for e in events:
        series[e['cursor']['updates']] = e
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    acc = json.loads(args.acceptance.read_text())
    v2 = acc['acceptance_v2']
    fb = load_series(args.run / 'dual_fallback' / 'PROGRESS.jsonl')
    fla = load_series(args.run / 'dual_fla' / 'PROGRESS.jsonl')
    common = sorted(set(fb) & set(fla))
    cap = v2['dual_updates']
    require(len(common) >= cap, 'DUAL_SERIES_SHORT:%d' % len(common))
    common = common[:cap]
    warmup = v2['dual_warmup']
    ce_band = v2['ce_band']
    ce_final = v2['ce_final_band']
    recall_band = v2['recall_band']
    worst = 0.0
    worst_at = None
    for u in common:
        if u < warmup:
            continue
        a, b = fb[u]['metrics']['mean_ce'], fla[u]['metrics']['mean_ce']
        rel = abs(a - b) / max(abs(a), 1e-12)
        if rel > worst:
            worst, worst_at = rel, u
    final_from = cap - v2['final_window']
    fb_final = [fb[u]['metrics'] for u in common if u >= final_from]
    fla_final = [fla[u]['metrics'] for u in common if u >= final_from]
    fb_ce = sum(m['mean_ce'] for m in fb_final) / len(fb_final)
    fla_ce = sum(m['mean_ce'] for m in fla_final) / len(fla_final)
    recall_diffs = []
    for cls in range(4):
        rb = fb_final[-1]['action_recall'][cls]
        rf = fla_final[-1]['action_recall'][cls]
        if rb is None or rf is None:
            recall_diffs.append(None)
        else:
            recall_diffs.append(abs(rb - rf))
    gates = dict(
        updates_compared=len(common), warmup=warmup,
        ce_worst_relative=worst, ce_worst_at=worst_at,
        ce_band_pass=worst <= ce_band,
        final_mean_ce=dict(fallback=fb_ce, fla=fla_ce,
                           relative_diff=abs(fb_ce - fla_ce) / max(fb_ce, 1e-12)),
        ce_final_pass=abs(fb_ce - fla_ce) / max(fb_ce, 1e-12) <= ce_final,
        recall_diffs=recall_diffs,
        recall_pass=all(d is None or d <= recall_band for d in recall_diffs),
    )
    gates['status'] = 'PASS' if (gates['ce_band_pass'] and gates['ce_final_pass']
                                 and gates['recall_pass']) else 'FAIL'
    result = dict(unix=time.time(), gates=gates, thresholds=v2, fail_preserved=True)
    with (args.run / 'DUAL_COMPARISON.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(gates, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
