from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from .backends import ExpectationBackend
from .circuit import Circuit
from .cpt import CPTPath, cpt_estimate, enumerate_cpt_paths
from .pauli import PauliString


@dataclass(frozen=True)
class QueppResult:
    max_order: int
    ideal_cpt_value: float
    noisy_target_value: float
    noisy_low_order_value: float
    residual_noisy_value: float
    eta: float
    mitigated_value: float
    eta_values: tuple[float, ...]
    path_count: int
    measured_path_count: int
    unique_measured_circuits: int
    paths: tuple[CPTPath, ...]


def aggregate_eta(
    eta_values: Sequence[float],
    strategy: str = "median",
    weights: Sequence[float] | None = None,
) -> float:
    if not eta_values:
        raise ValueError("At least one non-zero Clifford path is required to estimate eta.")

    if strategy == "median":
        return float(median(eta_values))
    if strategy == "mean":
        return float(sum(eta_values) / len(eta_values))
    if strategy == "weighted_mean":
        if weights is None:
            raise ValueError("weights are required for the weighted_mean eta strategy.")
        denominator = float(sum(weights))
        if abs(denominator) < 1e-15:
            raise ValueError("Signed weights sum to zero; weighted_mean eta is undefined.")
        numerator = sum(weight * eta for weight, eta in zip(weights, eta_values))
        return float(numerator / denominator)
    raise ValueError(f"Unsupported eta aggregation strategy: {strategy}")


def run_order_based_quepp(
    circuit: Circuit,
    observable: PauliString,
    backend: ExpectationBackend,
    max_order: int,
    eta_strategy: str = "median",
) -> QueppResult:
    paths = enumerate_cpt_paths(circuit, observable, max_order=max_order)
    ideal_value = cpt_estimate(paths)
    noisy_target = float(backend.estimate(circuit, observable))

    noisy_cache: dict[tuple, float] = {}
    noisy_low_order = 0.0
    eta_values: list[float] = []
    eta_weights: list[float] = []
    measured_paths = 0

    for path in paths:
        if path.ideal_expectation == 0.0:
            continue
        measured_paths += 1
        signature = path.clifford_circuit.signature()
        if signature not in noisy_cache:
            noisy_cache[signature] = float(backend.estimate(path.clifford_circuit, observable))
        noisy_value = noisy_cache[signature]
        noisy_low_order += path.weight * noisy_value
        eta_values.append(noisy_value / path.ideal_expectation)
        eta_weights.append(path.weight * path.ideal_expectation)

    eta = aggregate_eta(eta_values, strategy=eta_strategy, weights=eta_weights)
    if abs(eta) < 1e-15:
        raise ValueError("Estimated eta is too close to zero.")

    residual_noisy = noisy_target - noisy_low_order
    mitigated = ideal_value + (residual_noisy / eta)

    return QueppResult(
        max_order=max_order,
        ideal_cpt_value=ideal_value,
        noisy_target_value=noisy_target,
        noisy_low_order_value=noisy_low_order,
        residual_noisy_value=residual_noisy,
        eta=eta,
        mitigated_value=float(mitigated),
        eta_values=tuple(float(value) for value in eta_values),
        path_count=len(paths),
        measured_path_count=measured_paths,
        unique_measured_circuits=len(noisy_cache),
        paths=tuple(paths),
    )
