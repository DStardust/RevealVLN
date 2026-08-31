"""Execution-equivalent accelerated fitting for the sealed MF3ZM-CAR model.

The scientific model, losses, hard decision rule, constraints, folds, seeds,
and optimizer are unchanged.  This implementation only hoists constant index
tensors out of the optimization loop and evaluates all leave-one-scene
constraints in one tensor operation per domain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from revealnav_mf3.car import (
    build_model,
    event_domain_weights,
    straight_through_gate,
    utility_weighted_preference_loss,
)
from revealnav_mf3.car_selection import (
    REPRESENTATIONS,
    RISK_MODES,
    _forward,
    _standardize,
    _state_hash,
    _tensor_inputs,
    _validate_inputs,
)
from revealnav_mf3.nested_selection import (
    CATASTROPHIC_THRESHOLD,
    NestedSelectionError,
)


ProgressCallback = Callable[[int, int, int, int], None]


def fit_car_ensemble_fast(
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    episodes: np.ndarray,
    *,
    weight_decay: float,
    seeds: Sequence[int],
    learning_rate: float,
    dual_learning_rate: float,
    training_steps: int,
    dual_cap: float,
    representation: str = "semantic",
    risk_mode: str = "hard",
    scene_constraint: bool = True,
    use_cuda: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[torch.nn.Module], list[str], list[dict]]:
    """Fit the sealed CAR ensemble with batched constant constraints."""

    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    episodes = np.asarray([str(value) for value in episodes])
    _validate_inputs(inputs, len(target), representation)
    seeds = tuple(int(value) for value in seeds)
    if (
        len(target) == 0
        or not np.isfinite(target).all()
        or not len(target) == len(scenes) == len(datasets) == len(episodes)
        or float(weight_decay) not in (0.0001, 0.001, 0.01)
        or len(seeds) != 3
        or learning_rate <= 0
        or dual_learning_rate <= 0
        or training_steps < 1
        or dual_cap <= 0
        or representation not in REPRESENTATIONS
        or risk_mode not in RISK_MODES
    ):
        raise ValueError("invalid CAR training configuration")

    event_weights = event_domain_weights(datasets)
    field = {
        "semantic": "policy",
        "engineered_28d": "engineered",
        "policy_only": "policy_only",
    }[representation]
    mean, scale = _standardize(inputs[field], event_weights)
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    tensors = _tensor_inputs(inputs, device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(
        event_weights, dtype=torch.float32, device=device
    )
    catastrophic_tensor = (
        target_tensor <= float(CATASTROPHIC_THRESHOLD)
    ).to(torch.float32)

    domains = sorted(set(datasets))
    domain_state: dict[str, dict] = {}
    for domain in domains:
        indices = np.flatnonzero(datasets == domain)
        local_scenes = scenes[indices]
        scene_values = sorted(set(local_scenes))
        subset_matrix = np.stack(
            [local_scenes != scene for scene in scene_values], axis=0
        )
        if not subset_matrix.any(axis=1).all():
            raise ValueError("CAR scene subset is empty")
        domain_state[domain] = {
            "indices_numpy": indices,
            "indices": torch.as_tensor(
                indices, dtype=torch.long, device=device
            ),
            "weights": torch.full(
                (len(indices),), 1.0 / len(indices),
                dtype=torch.float32, device=device,
            ),
            "base_rate": float(np.mean(
                target[indices] <= float(CATASTROPHIC_THRESHOLD)
            )),
            "scene_values": scene_values,
            "scene_subsets": torch.as_tensor(
                subset_matrix, dtype=torch.float32, device=device
            ),
        }

    models: list[torch.nn.Module] = []
    initial_hashes: list[str] = []
    diagnostics: list[dict] = []
    callback_stride = max(1, int(training_steps) // 10)
    for seed_index, seed in enumerate(seeds):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = build_model(representation).to(device)
        if representation in {"semantic", "engineered_28d"}:
            setter = (
                model.set_policy_standardization
                if representation == "semantic"
                else model.set_standardization
            )
            setter(mean, scale)
        else:
            model.set_standardization(mean, scale)
        initial_hashes.append(_state_hash(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        risk_duals = {
            domain: torch.zeros((), dtype=torch.float32, device=device)
            for domain in domains
        }
        utility_duals = {
            domain: torch.zeros((), dtype=torch.float32, device=device)
            for domain in domains
        }
        scene_duals = {
            domain: torch.zeros(
                len(domain_state[domain]["scene_values"]),
                dtype=torch.float32, device=device,
            )
            for domain in domains
        }
        zero_risk_steps = {
            domain: torch.zeros((), dtype=torch.int64, device=device)
            for domain in domains
        }
        zero_utility_steps = {
            domain: torch.zeros((), dtype=torch.int64, device=device)
            for domain in domains
        }
        zero_scene_steps = {
            domain: torch.zeros(
                len(domain_state[domain]["scene_values"]),
                dtype=torch.int64, device=device,
            )
            for domain in domains
        }

        model.train()
        if progress_callback is not None:
            progress_callback(seed_index, seed, 0, int(training_steps))
        for step in range(int(training_steps)):
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, tensors, representation)
            objective = utility_weighted_preference_loss(
                logits, target_tensor, weight_tensor
            )
            step_constraints: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for domain in domains:
                state = domain_state[domain]
                local_index = state["indices"]
                d_logits = logits[local_index]
                d_target = target_tensor[local_index]
                d_catastrophic = catastrophic_tensor[local_index]
                d_weights = state["weights"]
                surrogate, hard, probability = straight_through_gate(d_logits)

                if risk_mode != "none":
                    selected = surrogate if risk_mode == "hard" else probability
                    selected_mass = torch.sum(d_weights * selected)
                    if risk_mode == "hard":
                        nonzero = torch.sum(d_weights * hard) > 0
                    else:
                        nonzero = torch.ones((), dtype=torch.bool, device=device)
                    safe_mass = torch.where(
                        nonzero, selected_mass, torch.ones_like(selected_mass)
                    )
                    risk_value = (
                        torch.sum(d_weights * selected * d_catastrophic)
                        / safe_mass - float(state["base_rate"])
                    )
                    risk_constraint = torch.where(
                        nonzero, risk_value, torch.sum(d_logits) * 0.0
                    )
                    zero_risk_steps[domain] += (~nonzero).to(torch.int64)
                    objective = objective + risk_duals[domain] * risk_constraint
                else:
                    risk_constraint = torch.zeros(
                        (), dtype=torch.float32, device=device
                    )

                contribution = d_weights * surrogate * d_target
                utility_constraint = -torch.sum(contribution)
                hard_any = torch.any(hard > 0)
                zero_utility_steps[domain] += (~hard_any).to(torch.int64)
                objective = (
                    objective + utility_duals[domain] * utility_constraint
                )

                if scene_constraint:
                    subsets = state["scene_subsets"]
                    scene_constraints = -(subsets @ contribution)
                    selected_by_subset = subsets @ hard
                    zero_scene_steps[domain] += (
                        selected_by_subset <= 0
                    ).to(torch.int64)
                    objective = objective + torch.dot(
                        scene_duals[domain], scene_constraints
                    )
                else:
                    scene_constraints = torch.empty(
                        0, dtype=torch.float32, device=device
                    )
                step_constraints[domain] = (
                    risk_constraint, utility_constraint, scene_constraints
                )

            penalty = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in model.parameters():
                penalty = penalty + torch.sum(parameter ** 2)
            objective = objective + float(weight_decay) * penalty
            if not bool(torch.isfinite(objective)):
                raise NestedSelectionError("CAR optimization became non-finite")
            objective.backward()
            optimizer.step()

            for domain in domains:
                risk_constraint, utility_constraint, scene_constraints = (
                    step_constraints[domain]
                )
                if risk_mode != "none":
                    risk_duals[domain] = torch.clamp(
                        risk_duals[domain]
                        + float(dual_learning_rate) * risk_constraint.detach(),
                        min=0.0, max=float(dual_cap),
                    )
                utility_duals[domain] = torch.clamp(
                    utility_duals[domain]
                    + float(dual_learning_rate) * utility_constraint.detach(),
                    min=0.0, max=float(dual_cap),
                )
                if scene_constraint:
                    scene_duals[domain] = torch.clamp(
                        scene_duals[domain]
                        + float(dual_learning_rate) * scene_constraints.detach(),
                        min=0.0, max=float(dual_cap),
                    )
            completed = step + 1
            if (
                progress_callback is not None
                and (completed % callback_stride == 0
                     or completed == int(training_steps))
            ):
                progress_callback(
                    seed_index, seed, completed, int(training_steps)
                )

        model.eval()
        with torch.no_grad():
            final_logits = _forward(model, tensors, representation)
            final_loss = utility_weighted_preference_loss(
                final_logits, target_tensor, weight_tensor
            )
        final_logits_np = final_logits.detach().cpu().numpy()
        hard = final_logits_np > 0.0
        models.append(model.cpu())
        zero_risk = {
            domain: int(zero_risk_steps[domain].detach().cpu())
            for domain in domains
            if int(zero_risk_steps[domain].detach().cpu()) > 0
        }
        zero_scene: dict[str, int] = {}
        for domain in domains:
            domain_zero = int(zero_utility_steps[domain].detach().cpu())
            if domain_zero > 0:
                zero_scene[f"{domain}:domain"] = domain_zero
            values = zero_scene_steps[domain].detach().cpu().tolist()
            for scene, value in zip(
                domain_state[domain]["scene_values"], values, strict=True
            ):
                if int(value) > 0:
                    zero_scene[f"{domain}:{scene}"] = int(value)
        diagnostics.append({
            "seed": seed,
            "device": str(device),
            "preference_loss": float(final_loss.detach().cpu()),
            "risk_mode": risk_mode,
            "scene_constraint": bool(scene_constraint),
            "zero_risk_steps": zero_risk,
            "zero_scene_steps": zero_scene,
            "hard_selected_by_domain": {
                domain: int(hard[domain_state[domain]["indices_numpy"]].sum())
                for domain in domains
            },
            "dual_variables": {
                "risk": {
                    domain: float(risk_duals[domain].detach().cpu())
                    for domain in domains
                },
                "utility": {
                    domain: float(utility_duals[domain].detach().cpu())
                    for domain in domains
                },
                "scene_max": float(max(
                    (
                        float(torch.max(value).detach().cpu())
                        for value in scene_duals.values() if len(value) > 0
                    ),
                    default=0.0,
                )),
            },
            "ungated_event_catastrophic_rate": {
                domain: float(domain_state[domain]["base_rate"])
                for domain in domains
            },
        })
    return models, initial_hashes, diagnostics


__all__ = ["fit_car_ensemble_fast"]
