#!/usr/bin/env python3
"""Build the fixed, outcome-blind MF3ZQ oracle-headroom population.

Only the previously sealed 80-event independent visual-review labels and the
corresponding causal observation metadata are read.  No task metric, reward,
CAR result, or public split is opened.  Missing option-specific DEC bindings
are recorded as unsupported instead of being guessed.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.oracle_revealskill_schema import K_STABILITY  # noqa: E402
from revealnav_mf3.oracle_headroom_protocol import (  # noqa: E402
    AUDIT_PATH,
    EVENTS,
    FORMAL_MF3ZP,
    OUTPUT,
    POPULATION_PATH,
    PUBLIC_CLOSED,
    SOURCE_OBSERVATIONS,
    VISUAL_LABELS,
    VISUAL_MANIFEST,
    VISUAL_PROTOCOL,
    inventory,
    sha256_file,
)


SELECTION_LABEL_COUNT = 80
DOMAIN_COUNTS = {"R2R": 40, "RxR": 40}


class PopulationError(RuntimeError):
    pass


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PopulationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PopulationError(f"JSON object required at {path}:{number}")
        rows.append(value)
    if not rows:
        raise PopulationError(f"empty JSONL: {path}")
    return rows


FORBIDDEN = {
    "reward", "success", "spl", "ndtw", "sdtw", "utility", "delta_utility",
    "outcome", "catastrophe", "target", "future", "future_frame",
    "future_candidate_set", "correct_action", "best_action", "pose", "navmesh",
    "car_result", "ree_result", "prediction",
}


def reject_outcomes(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in FORBIDDEN or lowered.startswith(("future_", "outcome_", "reward_", "treatment_")):
                raise PopulationError(f"forbidden outcome field {path}.{key}")
            reject_outcomes(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_outcomes(child, f"{path}[{index}]")


def _event_index() -> dict[str, dict]:
    events = _read_jsonl(EVENTS)
    result = {str(row["event_id"]): row for row in events}
    if len(result) != len(events):
        raise PopulationError("duplicate source reveal event identity")
    return result


def _option_binding(graph: list[dict], options: tuple[str, ...], roles: Mapping[str, str]) -> dict[str, dict[str, list[str]]]:
    """Use only explicit exact bindings; never infer a correct action.

    The old independent visual review usually left ``decisive_for`` empty.  A
    natural-language text that is not exactly an opaque candidate ID cannot be
    safely aligned, so the corresponding event remains unsupported.
    """

    option_set = set(options)
    result = {option: {"decisive": [], "prerequisite": []} for option in options}
    for constraint in graph:
        cid = str(constraint["constraint_id"])
        role = roles.get(cid)
        for raw_option in constraint.get("decisive_for", []):
            candidate = str(raw_option)
            if candidate in option_set:
                if role == "DEC_REQUIRED":
                    result[candidate]["decisive"].append(cid)
                elif role == "PREREQUISITE_ONLY":
                    result[candidate]["prerequisite"].append(cid)
    return result


def _factor_sequences(label: Mapping[str, object], steps: list[int]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for cid, item in label["constraints"].items():
        values = item.get("factor_by_step", [])
        by_step = {int(row["step"]): row for row in values}
        if item["dec_role"] in {"DEC_REQUIRED", "PREREQUISITE_ONLY"}:
            if set(by_step) != set(steps):
                raise PopulationError(f"incomplete factors for {label['event_id']}:{cid}")
            output[str(cid)] = [
                {
                    "step": step,
                    "instantiated": bool(by_step[step]["instantiated"]),
                    "distinguishable": bool(by_step[step]["distinguishable"]),
                    "resolved": bool(by_step[step]["resolved"]),
                }
                for step in steps
            ]
    return output


def _build_row(label: Mapping[str, object], event: Mapping[str, object], blacklist: set[str]) -> dict:
    reject_outcomes(label)
    reject_outcomes(event)
    if label.get("public_split_access") != PUBLIC_CLOSED:
        raise PopulationError(f"source label opened a public split: {label['event_id']}")
    if str(label["event_id"]) != str(event["event_id"]):
        raise PopulationError("label/event identity mismatch")
    if str(label["scene_id"]) != str(event["scene_id"]) or str(label["episode_id"]) != str(event["episode_id"]):
        raise PopulationError("label/event context mismatch")
    if str(label["scene_id"]) in blacklist:
        raise PopulationError("consumed confirmation scene entered MF3ZQ population")
    options = tuple(str(value) for value in event.get("option_ids", ()))
    if not options or len(options) != len(set(options)):
        raise PopulationError("event has no unique opaque options")
    prefixes = label.get("prefix_sources")
    if not isinstance(prefixes, list) or not prefixes:
        raise PopulationError("label has no causal prefix sources")
    steps = [int(prefix["step"]) for prefix in prefixes]
    decision_step = int(label["decision_step"])
    if steps != sorted(steps) or steps[-1] != decision_step or any(step > decision_step for step in steps):
        raise PopulationError("non-causal or misaligned prefix window")
    if tuple(str(value) for value in prefixes[-1]["candidate_ids"]) != options:
        raise PopulationError("option aliases differ from reviewed current candidates")
    graph = list(label.get("constraint_graph", ()))
    roles = {str(cid): str(value["dec_role"]) for cid, value in label["constraints"].items()}
    if len(roles) != len(graph) + len(label.get("independent_missing_constraints", [])):
        # Missing atoms are intentionally not silently merged into the frozen
        # graph.  The event can still be represented, but is unsupported for
        # option-specific execution until a formal annotation binds it.
        pass
    dec_ids = tuple(cid for cid, role in roles.items() if role == "DEC_REQUIRED")
    pre_ids = tuple(cid for cid, role in roles.items() if role == "PREREQUISITE_ONLY")
    bindings = _option_binding(graph, options, roles)
    unsupported_reasons = []
    if not dec_ids:
        unsupported_reasons.append("no_DEC_REQUIRED_constraints")
    if any(not bindings[option]["decisive"] for option in options):
        unsupported_reasons.append("missing_explicit_option_DEC_binding")
    if label.get("independent_missing_constraints"):
        unsupported_reasons.append("unbound_independent_missing_constraint")
    observation_dir = SOURCE_OBSERVATIONS / str(label["dataset"]) / f"ep_{label['episode_id']}" / "attempt_001"
    required_files = (observation_dir / "RUN_SUMMARY.json", observation_dir / "base_trace.jsonl", observation_dir / "causal_prefix_records.jsonl")
    if any(not path.is_file() or path.is_symlink() for path in required_files):
        unsupported_reasons.append("native_observation_files_missing")
    factor_sequences = _factor_sequences(label, steps)
    return {
        "schema_version": "revealnav-mf3zq-oracle-headroom-population/1",
        "event_id": str(label["event_id"]),
        "dataset": str(label["dataset"]),
        "scene_id": str(label["scene_id"]),
        "episode_id": str(label["episode_id"]),
        "instruction": str(label["instruction"]),
        "decision_step": decision_step,
        "prefix_steps": steps,
        "option_ids": list(options),
        "candidate_aliases_at_decision": list(options),
        "observation_dir": str(observation_dir.relative_to(ROOT)),
        "constraint_graph": graph,
        "constraint_roles": roles,
        "factor_sequences": factor_sequences,
        "decisive_constraint_ids": list(dec_ids),
        "prerequisite_constraint_ids": list(pre_ids),
        "option_bindings": bindings,
        "independent_missing_constraints": list(label.get("independent_missing_constraints", [])),
        "qwen_or_human_label_source": "independent_visual_review_not_human_not_gold",
        "support": {
            "option_specific_dec_binding_complete": not any(not bindings[option]["decisive"] for option in options),
            "control_backed_returnability_available": False,
            "legal_oracle_continuation_supported": not unsupported_reasons,
            "unsupported_reasons": sorted(set(unsupported_reasons)),
        },
    }


def _write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise PopulationError(f"refusing to overwrite existing MF3ZQ artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise PopulationError(f"stale partial artifact: {part}")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def build() -> dict:
    labels = _read_jsonl(VISUAL_LABELS)
    if len(labels) != SELECTION_LABEL_COUNT:
        raise PopulationError(f"expected exactly {SELECTION_LABEL_COUNT} visual-review events")
    if Counter(str(row["dataset"]) for row in labels) != Counter(DOMAIN_COUNTS):
        raise PopulationError("domain allocation drift")
    if len({(str(row["dataset"]), str(row["episode_id"])) for row in labels}) != SELECTION_LABEL_COUNT:
        raise PopulationError("MF3ZQ requires one unique episode per selected event")
    events = _event_index()
    formal = _read_json(FORMAL_MF3ZP)
    blacklist = {str(value) for value in formal.get("consumed_confirmation_blacklist", [])}
    rows = []
    for label in labels:
        event = events.get(str(label["event_id"]))
        if event is None:
            raise PopulationError(f"visual event missing from sealed reveal events: {label['event_id']}")
        rows.append(_build_row(label, event, blacklist))
    rows.sort(key=lambda row: (row["dataset"], row["scene_id"], row["episode_id"], int(row["decision_step"]), row["event_id"]))
    if len({row["event_id"] for row in rows}) != SELECTION_LABEL_COUNT:
        raise PopulationError("population event identity collision")
    lines = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    _write_new(POPULATION_PATH, lines)
    role_counts = Counter(role for row in rows for role in row["constraint_roles"].values())
    unsupported = [row for row in rows if not row["support"]["legal_oracle_continuation_supported"]]
    scenes = sorted({row["scene_id"] for row in rows})
    audit = {
        "schema_version": "revealnav-mf3zq-oracle-headroom-population-audit/1",
        "status": "MF3ZQ_POPULATION_AUDIT_PASS",
        "revision": "mf3zq_oracle_revealskill_headroom_v1",
        "events": len(rows),
        "unique_episodes": len({(row["dataset"], row["episode_id"]) for row in rows}),
        "raw_mp3d_scenes": len(scenes),
        "scene_ids": scenes,
        "domain_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "role_counts": dict(sorted(role_counts.items())),
        "unsupported_episode_count": len(unsupported),
        "unsupported_event_ids": [row["event_id"] for row in unsupported],
        "unsupported_reasons": dict(sorted(Counter(reason for row in unsupported for reason in row["support"]["unsupported_reasons"]).items())),
        "active_decisive_fraction": float(role_counts["DEC_REQUIRED"] / sum(role_counts.values())) if role_counts else 0.0,
        "current_relevant_fraction": float((role_counts["DEC_REQUIRED"] + role_counts["PREREQUISITE_ONLY"]) / sum(role_counts.values())) if role_counts else 0.0,
        "decomposition": {
            "DEC_REQUIRED": role_counts["DEC_REQUIRED"],
            "PREREQUISITE_ONLY": role_counts["PREREQUISITE_ONLY"],
            "FUTURE_NOT_RELEVANT": role_counts["FUTURE_NOT_RELEVANT"],
            "INCORRECT": role_counts["INCORRECT"],
            "REDUNDANT": role_counts["REDUNDANT"],
            "missing_DEC_constraints": sum(len(row["independent_missing_constraints"]) for row in rows),
        },
        "source_inventory": {
            "visual_labels": inventory(VISUAL_LABELS),
            "visual_manifest": inventory(VISUAL_MANIFEST),
            "visual_protocol": inventory(VISUAL_PROTOCOL),
            "reveal_events": inventory(EVENTS),
            "formal_mf3zp_protocol": inventory(FORMAL_MF3ZP),
            "population": inventory(POPULATION_PATH),
        },
        "outcome_payload_read": False,
        "qwen_reads": 0,
        "qwen_calls": 0,
        "public_split_access": dict(PUBLIC_CLOSED),
        "checkpoint_generated": False,
    }
    _write_new(AUDIT_PATH, json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return audit


def main() -> int:
    try:
        result = build()
    except (OSError, KeyError, TypeError, ValueError, PopulationError) as error:
        print(f"MF3ZQ_POPULATION_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
