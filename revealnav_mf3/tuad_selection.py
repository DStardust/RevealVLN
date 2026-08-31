"""Fixed scene-OOF selection and scientific gates for MF3ZN-TUAD v1.

There is deliberately no hyperparameter, architecture, seed, or threshold
selection in this module.  It only assigns whole-scene folds, constructs exact
budget controls, and evaluates the pre-registered one-shot PASS/FAIL criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np

from .nested_selection import deterministic_scene_folds


FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED = True
TUAD_OUTER_FOLDS = 5
TUAD_FOLD_SALT = "mf3zn_tuad_v1_raw_scene_oof"
REQUIRED_DOMAINS = ("R2R", "RxR")
REQUIRED_POLICIES = (
    "TUAD-full",
    "current-only",
    "temporal-no-UAD-supervision",
    "oracle-UAD",
    "runner-only-support",
    "frozen-native",
    "matched-high-proposal-score",
    "matched-low-native-margin",
    "matched-random",
)


@dataclass(frozen=True)
class PolicyOutcomes:
    """Per-event native-relative outcome of one fixed policy."""

    utility: np.ndarray
    selected: np.ndarray
    catastrophic: np.ndarray

    def arrays(self, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        utility = np.asarray(self.utility, dtype=np.float64)
        selected = np.asarray(self.selected, dtype=bool)
        catastrophic = np.asarray(self.catastrophic, dtype=bool)
        if any(value.ndim != 1 or len(value) != length for value in (
            utility, selected, catastrophic
        )):
            raise ValueError("policy outcome arrays have the wrong shape")
        if not np.isfinite(utility).all():
            raise ValueError("policy utilities contain non-finite values")
        if np.any(~selected & ((utility != 0.0) | catastrophic)):
            raise ValueError("native abstentions must have zero utility and no catastrophe")
        return utility, selected, catastrophic


def assign_tuad_scene_folds(
    scenes: Sequence[object],
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign the single fixed five-fold raw-MP3D-scene partition."""

    return deterministic_scene_folds(
        scenes, TUAD_OUTER_FOLDS, salt=TUAD_FOLD_SALT
    )


def validate_lattice_fold_integrity(
    scenes: Sequence[object],
    episodes: Sequence[object],
    lattice_ids: Sequence[object],
    folds: Sequence[int],
) -> None:
    """Fail if a scene, episode, or action-lattice arm crosses folds."""

    arrays = [np.asarray([str(value) for value in values]) for values in (
        scenes, episodes, lattice_ids
    )]
    fold = np.asarray(folds, dtype=np.int64)
    if not arrays or any(value.ndim != 1 for value in arrays) or fold.ndim != 1:
        raise ValueError("invalid fold-integrity inputs")
    length = len(fold)
    if length == 0 or any(len(value) != length for value in arrays):
        raise ValueError("fold-integrity inputs have mismatched lengths")
    if set(fold.tolist()) != set(range(TUAD_OUTER_FOLDS)):
        raise ValueError("TUAD requires a complete five-fold OOF assignment")
    for name, values in zip(("scene", "episode", "lattice"), arrays):
        assignments: dict[str, set[int]] = {}
        for identity, value in zip(values, fold, strict=True):
            assignments.setdefault(str(identity), set()).add(int(value))
        split = sorted(key for key, value in assignments.items() if len(value) != 1)
        if split:
            raise ValueError(f"{name} crosses folds: {split[:3]}")


def _stable_random_key(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def matched_budget_baselines(
    full_selected: Sequence[bool],
    proposal_score: Sequence[float],
    native_margin: Sequence[float],
    folds: Sequence[int],
    datasets: Sequence[object],
    identities: Sequence[object],
    *,
    random_seed: int = 20260831,
) -> dict[str, np.ndarray]:
    """Create the three fixed exact fold×domain intervention-budget controls."""

    selected = np.asarray(full_selected, dtype=bool)
    score = np.asarray(proposal_score, dtype=np.float64)
    margin = np.asarray(native_margin, dtype=np.float64)
    fold = np.asarray(folds, dtype=np.int64)
    domain = np.asarray([str(value) for value in datasets])
    identity = np.asarray([str(value) for value in identities])
    length = len(selected)
    if length == 0 or any(
        value.ndim != 1 or len(value) != length
        for value in (score, margin, fold, domain, identity)
    ):
        raise ValueError("invalid matched-budget inputs")
    if not np.isfinite(score).all() or not np.isfinite(margin).all():
        raise ValueError("non-finite baseline ranking feature")
    if len(set(identity.tolist())) != length:
        raise ValueError("baseline event identities must be unique")
    masks = {
        "matched-high-proposal-score": np.zeros(length, dtype=bool),
        "matched-low-native-margin": np.zeros(length, dtype=bool),
        "matched-random": np.zeros(length, dtype=bool),
    }
    for fold_value in sorted(set(fold.tolist())):
        for domain_value in sorted(set(domain.tolist())):
            stratum = np.flatnonzero(
                (fold == fold_value) & (domain == domain_value)
            )
            budget = int(selected[stratum].sum())
            if budget == 0:
                continue
            high = sorted(
                stratum.tolist(), key=lambda index: (-score[index], identity[index])
            )
            low = sorted(
                stratum.tolist(), key=lambda index: (margin[index], identity[index])
            )
            random = sorted(
                stratum.tolist(),
                key=lambda index: (
                    _stable_random_key(random_seed, identity[index]), identity[index]
                ),
            )
            masks["matched-high-proposal-score"][high[:budget]] = True
            masks["matched-low-native-margin"][low[:budget]] = True
            masks["matched-random"][random[:budget]] = True
    validate_exact_fold_domain_budgets(selected, masks, fold, domain)
    return masks


def validate_exact_fold_domain_budgets(
    reference: Sequence[bool],
    baselines: Mapping[str, Sequence[bool]],
    folds: Sequence[int],
    datasets: Sequence[object],
) -> None:
    reference = np.asarray(reference, dtype=bool)
    fold = np.asarray(folds, dtype=np.int64)
    domain = np.asarray([str(value) for value in datasets])
    if reference.ndim != 1 or fold.shape != reference.shape or domain.shape != reference.shape:
        raise ValueError("invalid exact-budget arrays")
    for name, values in baselines.items():
        mask = np.asarray(values, dtype=bool)
        if mask.shape != reference.shape:
            raise ValueError(f"{name} has the wrong budget-mask shape")
        for fold_value in sorted(set(fold.tolist())):
            for domain_value in sorted(set(domain.tolist())):
                stratum = (fold == fold_value) & (domain == domain_value)
                if int(mask[stratum].sum()) != int(reference[stratum].sum()):
                    raise ValueError(
                        f"{name} violates fold={fold_value},domain={domain_value} budget"
                    )


def _catastrophic_rate(selected: np.ndarray, catastrophic: np.ndarray) -> float | None:
    count = int(selected.sum())
    return float(catastrophic[selected].mean()) if count else None


def _policy_evidence(
    outcomes: PolicyOutcomes,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> dict:
    utility, selected, catastrophic = outcomes.arrays(len(scenes))
    by_domain = {}
    for domain in REQUIRED_DOMAINS:
        domain_mask = datasets == domain
        scene_values = sorted(set(scenes[domain_mask & selected].tolist()))
        total = float(utility[domain_mask].sum())
        leave_one = [
            float(utility[domain_mask & (scenes != scene)].sum())
            for scene in scene_values
        ]
        by_fold = {}
        for fold in range(TUAD_OUTER_FOLDS):
            stratum = domain_mask & (folds == fold)
            if not bool(stratum.any()):
                raise ValueError(
                    f"missing fold={fold},domain={domain} development stratum"
                )
            by_fold[str(fold)] = {
                "utility": float(utility[stratum].sum()),
                "selected": int(selected[stratum].sum()),
                "catastrophic_rate": _catastrophic_rate(
                    selected[stratum], catastrophic[stratum]
                ),
            }
        by_domain[domain] = {
            "utility": total,
            "selected": int(selected[domain_mask].sum()),
            "catastrophic_rate": _catastrophic_rate(
                selected[domain_mask], catastrophic[domain_mask]
            ),
            "minimum_leave_one_scene_out_total": (
                min(leave_one) if leave_one else None
            ),
            "folds": by_fold,
        }
    return {"by_domain": by_domain}


def scene_cluster_difference(
    left: Sequence[float],
    right: Sequence[float],
    scenes: Sequence[object],
    datasets: Sequence[object],
    *,
    replicates: int = 10_000,
    seed: int = 20260831,
) -> dict:
    """Raw-scene bootstrap of per-domain total-utility differences."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scene = np.asarray([str(value) for value in scenes])
    domain = np.asarray([str(value) for value in datasets])
    if (
        left.ndim != 1
        or left.shape != right.shape
        or scene.shape != left.shape
        or domain.shape != left.shape
        or len(left) == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or int(replicates) < 1
    ):
        raise ValueError("invalid scene-bootstrap inputs")
    rng = np.random.default_rng(int(seed))
    result = {}
    delta = left - right
    for domain_value in REQUIRED_DOMAINS:
        domain_scenes = np.unique(scene[domain == domain_value])
        if len(domain_scenes) == 0:
            raise ValueError(f"missing required domain {domain_value}")
        totals = np.empty(int(replicates), dtype=np.float64)
        indices = {
            value: np.flatnonzero((scene == value) & (domain == domain_value))
            for value in domain_scenes
        }
        for replicate in range(int(replicates)):
            sampled = rng.choice(domain_scenes, size=len(domain_scenes), replace=True)
            totals[replicate] = sum(float(delta[indices[value]].sum()) for value in sampled)
        result[domain_value] = {
            "observed": float(delta[domain == domain_value].sum()),
            "lower_95": float(np.quantile(totals, 0.025)),
            "upper_95": float(np.quantile(totals, 0.975)),
        }
    return result


def materialize_policy_outcomes(
    chosen_action: Sequence[int],
    exact_delta_utility: np.ndarray,
    catastrophic: np.ndarray,
    action_mask: np.ndarray,
    is_native: np.ndarray,
) -> PolicyOutcomes:
    """Map a native-inclusive chosen action to exact paired outcomes."""

    chosen = np.asarray(chosen_action, dtype=np.int64)
    utility = np.asarray(exact_delta_utility, dtype=np.float64)
    catastrophic = np.asarray(catastrophic, dtype=bool)
    action_mask = np.asarray(action_mask, dtype=bool)
    is_native = np.asarray(is_native, dtype=bool)
    if (
        utility.ndim != 2
        or utility.shape != catastrophic.shape
        or utility.shape != action_mask.shape
        or utility.shape != is_native.shape
        or chosen.ndim != 1
        or len(chosen) != len(utility)
        or not np.isfinite(utility).all()
        or np.any(is_native.sum(axis=1) != 1)
        or np.any(is_native & ~action_mask)
        or np.any(utility[is_native] != 0.0)
    ):
        raise ValueError("invalid exact action outcome lattice")
    if np.any((chosen < 0) | (chosen >= utility.shape[1])):
        raise ValueError("chosen action index is outside the sealed lattice")
    row = np.arange(len(chosen))
    if np.any(~action_mask[row, chosen]):
        raise ValueError("policy chose a padded or non-executable action")
    selected = ~is_native[row, chosen]
    realized_utility = np.where(selected, utility[row, chosen], 0.0)
    realized_catastrophic = selected & catastrophic[row, chosen]
    return PolicyOutcomes(realized_utility, selected, realized_catastrophic)


def assemble_development_policies(
    chosen_by_arm: Mapping[str, Sequence[int]],
    exact_delta_utility: np.ndarray,
    catastrophic: np.ndarray,
    action_mask: np.ndarray,
    is_native: np.ndarray,
    proposal_score: Sequence[float],
    native_margin: Sequence[float],
    folds: Sequence[int],
    datasets: Sequence[object],
    identities: Sequence[object],
) -> dict[str, PolicyOutcomes]:
    """Construct every fixed learned/native/simple policy on one exact lattice."""

    learned = {
        "TUAD-full",
        "current-only",
        "temporal-no-UAD-supervision",
        "oracle-UAD",
        "runner-only-support",
    }
    if set(chosen_by_arm) != learned:
        raise ValueError(
            f"learned control set drift; missing={sorted(learned - set(chosen_by_arm))}, "
            f"extra={sorted(set(chosen_by_arm) - learned)}"
        )
    utility = np.asarray(exact_delta_utility, dtype=np.float64)
    catastrophic_array = np.asarray(catastrophic, dtype=bool)
    action_mask_array = np.asarray(action_mask, dtype=bool)
    native_array = np.asarray(is_native, dtype=bool)
    policies = {
        name: materialize_policy_outcomes(
            chosen, utility, catastrophic_array, action_mask_array, native_array
        )
        for name, chosen in chosen_by_arm.items()
    }
    rows = len(utility)
    frozen_chosen = native_array.astype(np.int64).argmax(axis=1)
    policies["frozen-native"] = materialize_policy_outcomes(
        frozen_chosen, utility, catastrophic_array, action_mask_array, native_array
    )
    baseline_masks = matched_budget_baselines(
        policies["TUAD-full"].selected,
        proposal_score,
        native_margin,
        folds,
        datasets,
        identities,
    )
    runner = np.empty(rows, dtype=np.int64)
    for row in range(rows):
        alternatives = np.flatnonzero(action_mask_array[row] & ~native_array[row])
        if len(alternatives) == 0:
            raise ValueError("simple baseline event has no sealed non-native action")
        runner[row] = alternatives[0]
    row_index = np.arange(rows)
    for name, selected in baseline_masks.items():
        selected = np.asarray(selected, dtype=bool)
        policies[name] = PolicyOutcomes(
            np.where(selected, utility[row_index, runner], 0.0),
            selected,
            selected & catastrophic_array[row_index, runner],
        )
    if set(policies) != set(REQUIRED_POLICIES):
        raise RuntimeError("assembled TUAD policy inventory drift")
    return policies


def evaluate_tuad_development(
    policies: Mapping[str, PolicyOutcomes],
    scenes: Sequence[object],
    datasets: Sequence[object],
    folds: Sequence[int],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260831,
) -> dict:
    """Apply the single frozen scientific gate without selecting a model."""

    missing = sorted(set(REQUIRED_POLICIES) - set(policies))
    extra = sorted(set(policies) - set(REQUIRED_POLICIES))
    if missing or extra:
        raise ValueError(f"policy arms drifted; missing={missing}, extra={extra}")
    scene = np.asarray([str(value) for value in scenes])
    domain = np.asarray([str(value) for value in datasets])
    fold = np.asarray(folds, dtype=np.int64)
    if scene.ndim != 1 or len(scene) == 0 or domain.shape != scene.shape or fold.shape != scene.shape:
        raise ValueError("invalid TUAD development population")
    if set(domain.tolist()) != set(REQUIRED_DOMAINS):
        raise ValueError("TUAD report must contain RxR and R2R separately")
    scene_assignment: dict[str, set[int]] = {}
    for value, fold_value in zip(scene, fold, strict=True):
        scene_assignment.setdefault(value, set()).add(int(fold_value))
    if any(len(value) != 1 for value in scene_assignment.values()):
        raise ValueError("a raw MP3D scene crosses outer folds")
    if set(fold.tolist()) != set(range(TUAD_OUTER_FOLDS)):
        raise ValueError("outer OOF predictions are incomplete")

    evidence = {
        name: _policy_evidence(value, scene, domain, fold)
        for name, value in policies.items()
    }
    arrays = {
        name: value.arrays(len(scene)) for name, value in policies.items()
    }
    failures: list[str] = []
    full = evidence["TUAD-full"]["by_domain"]
    for domain_value in REQUIRED_DOMAINS:
        item = full[domain_value]
        if not item["utility"] > 0.0:
            failures.append(f"{domain_value}:utility_nonpositive")
        if item["minimum_leave_one_scene_out_total"] is None or not (
            item["minimum_leave_one_scene_out_total"] > 0.0
        ):
            failures.append(f"{domain_value}:leave_one_scene_nonpositive")
        for fold_value, fold_item in item["folds"].items():
            if fold_item["utility"] < 0.0:
                failures.append(f"{domain_value}:fold_{fold_value}:utility_negative")

        simple_names = (
            "matched-high-proposal-score", "matched-low-native-margin"
        )
        strongest = max(
            simple_names,
            key=lambda name: evidence[name]["by_domain"][domain_value]["utility"],
        )
        comparator = evidence[strongest]["by_domain"][domain_value]
        if not item["utility"] > comparator["utility"]:
            failures.append(f"{domain_value}:not_above_strongest_simple_baseline")
        if item["catastrophic_rate"] is None or comparator["catastrophic_rate"] is None:
            failures.append(f"{domain_value}:undefined_catastrophic_rate")
        elif item["catastrophic_rate"] > comparator["catastrophic_rate"]:
            failures.append(f"{domain_value}:catastrophic_rate_above_{strongest}")

    contribution = {}
    for control, label in (
        ("current-only", "temporal"),
        ("temporal-no-UAD-supervision", "uad_reveal_expiry"),
    ):
        interval = scene_cluster_difference(
            arrays["TUAD-full"][0], arrays[control][0], scene, domain,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + (0 if control == "current-only" else 1),
        )
        contribution[label] = {"comparator": control, "domains": interval}
        for domain_value in REQUIRED_DOMAINS:
            if interval[domain_value]["observed"] <= 0.0:
                failures.append(f"{domain_value}:{label}_delta_nonpositive")
            if (
                control == "current-only"
                and interval[domain_value]["lower_95"] <= 0.0
            ):
                failures.append(f"{domain_value}:{label}_bootstrap_lower_nonpositive")

    return {
        "schema_version": "revealnav-mf3zn-tuad-development-result/1",
        "method_id": "mf3zn_tuad_v1",
        "selection_performed": False,
        "complete_outer_scene_oof": True,
        "status": "TUAD_DEVELOPMENT_PASS" if not failures else "TUAD_DEVELOPMENT_FAIL",
        "failures": failures,
        "policies": evidence,
        "contribution": contribution,
        "public_authorization": False,
    }


__all__ = [
    "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
    "PolicyOutcomes",
    "REQUIRED_POLICIES",
    "TUAD_OUTER_FOLDS",
    "assign_tuad_scene_folds",
    "assemble_development_policies",
    "evaluate_tuad_development",
    "materialize_policy_outcomes",
    "matched_budget_baselines",
    "scene_cluster_difference",
    "validate_exact_fold_domain_budgets",
    "validate_lattice_fold_integrity",
]
