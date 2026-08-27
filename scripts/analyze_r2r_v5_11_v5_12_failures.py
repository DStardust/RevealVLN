#!/usr/bin/env python3
"""Explain V5.11/V5.12 val-seen failures from already-opened paired traces."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "artifacts/evaluation"
OUT = EVALUATION / "mf2_r2r_v5_13_net_advantage"
PROTOCOL = EVALUATION / (
    "mf2_r2r_v5_12_reversible_dev_gate/R2R_V5_12_REVERSIBLE_DEV_PROTOCOL_V2.json"
)
VERSIONS = {
    "v5_11": EVALUATION / "mf2_r2r_v5_11_paired_seen_gate",
    "v5_12": EVALUATION / "mf2_r2r_v5_12_reversible_dev_gate",
}
BASELINE_ROOTS = (
    EVALUATION / "mf2_r2r_v5_11_temporal_diagnostic/runs",
    EVALUATION / "mf2_r2r_v5_11_fresh_activation_screen/runs",
)
HIGHER = {"success", "oracle_success", "spl", "ndtw", "sdtw"}
METRICS = (
    "success", "oracle_success", "spl", "ndtw", "sdtw",
    "distance_to_goal", "path_length", "steps_taken", "collisions",
)
ACTIVITY = (
    "effective_commit_interventions", "explore_decisions",
    "checkpointed_excursions", "backtrack_decisions", "successful_returns",
    "failed_returns",
)


def baseline_metrics(episode_id: str) -> dict:
    candidates = [
        root / f"shadow_ep_{episode_id}" for root in BASELINE_ROOTS
        if (root / f"shadow_ep_{episode_id}" / "RUN_SUMMARY.json").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous V5.11 baseline for episode {episode_id}")
    stats = list(candidates[0].glob(
        "etp_output/**/stats_ep_ckpt_270_val_seen_r0_w1.json"
    ))
    if len(stats) != 1:
        raise RuntimeError(f"ambiguous baseline metric file for {episode_id}")
    return json.loads(stats[0].read_text())[episode_id]


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def analyze_version(
    name: str, root: Path, episodes: list[str], seeds: list[int], baselines: dict,
) -> dict:
    runs = {}
    events = Counter()
    for path in root.glob("runs/*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        key = (str(row["episode_id"]), int(row["seed"]))
        if key in runs:
            raise RuntimeError(f"duplicate {name} paired run: {key}")
        runs[key] = row
        trace = path.with_name("controller_trace.jsonl")
        for line in trace.open():
            events[json.loads(line)["event"]] += 1
    expected = {(episode, seed) for episode in episodes for seed in seeds}
    if set(runs) != expected or any(row.get("status") != "PASS" for row in runs.values()):
        raise RuntimeError(f"{name} paired matrix is incomplete")
    per_episode = {}
    for episode in episodes:
        benefit = {}
        for metric in METRICS:
            direction = 1.0 if metric in HIGHER else -1.0
            benefit[metric] = sum(
                direction * (
                    float(runs[(episode, seed)]["metrics"][metric])
                    - float(baselines[episode][metric])
                ) for seed in seeds
            ) / len(seeds)
        activity = {
            key: sum(
                float(runs[(episode, seed)]["controller"][key]) for seed in seeds
            ) / len(seeds)
            for key in ACTIVITY
        }
        per_episode[episode] = {"benefit": benefit, "activity": activity}
    harmed = [episode for episode in episodes if per_episode[episode]["benefit"]["spl"] < 0]
    not_harmed = [episode for episode in episodes if episode not in harmed]
    intervention = [
        per_episode[episode]["activity"]["effective_commit_interventions"]
        + per_episode[episode]["activity"]["explore_decisions"]
        for episode in episodes
    ]
    spl = [per_episode[episode]["benefit"]["spl"] for episode in episodes]
    return {
        "paired_runs": len(runs),
        "paired_episodes": len(episodes),
        "mean_benefit": {
            metric: sum(per_episode[episode]["benefit"][metric] for episode in episodes)
            / len(episodes)
            for metric in METRICS
        },
        "total_activity": {
            key: int(sum(row["controller"][key] for row in runs.values()))
            for key in ACTIVITY
        },
        "trace_event_counts": dict(sorted(events.items())),
        "harm_diagnostics": {
            "episodes_with_negative_spl": len(harmed),
            "negative_spl_fraction": len(harmed) / len(episodes),
            "mean_interventions_harmed": (
                sum(
                    per_episode[episode]["activity"]["effective_commit_interventions"]
                    + per_episode[episode]["activity"]["explore_decisions"]
                    for episode in harmed
                ) / len(harmed) if harmed else 0.0
            ),
            "mean_interventions_not_harmed": (
                sum(
                    per_episode[episode]["activity"]["effective_commit_interventions"]
                    + per_episode[episode]["activity"]["explore_decisions"]
                    for episode in not_harmed
                ) / len(not_harmed) if not_harmed else 0.0
            ),
            "pearson_interventions_vs_spl_benefit": pearson(intervention, spl),
        },
        "per_episode": per_episode,
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    episodes = [str(row["episode_id"]) for row in protocol["selection"]]
    seeds = [int(seed) for seed in protocol["seeds"]]
    baselines = {episode: baseline_metrics(episode) for episode in episodes}
    versions = {
        name: analyze_version(name, root, episodes, seeds, baselines)
        for name, root in VERSIONS.items()
    }
    change = {
        metric: versions["v5_12"]["mean_benefit"][metric]
        - versions["v5_11"]["mean_benefit"][metric]
        for metric in METRICS
    }
    value = {
        "schema_version": "revealnav-r2r-v5.11-v5.12-failure-analysis/1",
        "status": "PASS",
        "scope": "already-opened val_seen development metrics and causal traces only",
        "versions": versions,
        "v5_12_minus_v5_11_benefit": change,
        "findings": [
            (
                "V5.11 direct interventions are not a viable main result: mean SPL "
                f"benefit is {versions['v5_11']['mean_benefit']['spl']:.6f}."
            ),
            (
                "V5.12 return mechanics mostly execute, but branch selection remains "
                f"harmful: {versions['v5_12']['total_activity']['successful_returns']} "
                "successful returns versus "
                f"{versions['v5_12']['total_activity']['failed_returns']} failed, while "
                f"mean SPL benefit is {versions['v5_12']['mean_benefit']['spl']:.6f}."
            ),
            (
                "Reversibility alone does not recover task performance: V5.12 minus "
                f"V5.11 mean SPL benefit changes by {change['spl']:.6f} and nDTW by "
                f"{change['ndtw']:.6f}."
            ),
            (
                "The next intervention should therefore be a causal pre-action veto "
                "trained on branch-level net advantage, while V5.6 remains the validated "
                "proposal backbone."
            ),
        ],
        "task_metrics_already_opened_for_method_development": True,
        "fresh_confirmation_claim": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2R_V5_11_V5_12_FAILURE_ANALYSIS.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    markdown = OUT / "R2R_V5_11_V5_12_FAILURE_ANALYSIS.md"
    markdown.write_text(
        "# V5.11 / V5.12 offline failure analysis\n\n"
        + "\n".join(f"- {finding}" for finding in value["findings"])
        + "\n\nThis analysis uses previously opened val-seen development results only; "
        "it makes no fresh-confirmation or paper-performance claim.\n"
    )
    print(json.dumps({
        "status": value["status"],
        "v5_11_spl": versions["v5_11"]["mean_benefit"]["spl"],
        "v5_12_spl": versions["v5_12"]["mean_benefit"]["spl"],
        "v5_12_minus_v5_11_spl": change["spl"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
