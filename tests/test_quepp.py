import math

from quepp import (
    Circuit,
    PauliString,
    RescalingNoiseBackend,
    cpt_estimate,
    enumerate_cpt_paths,
    exact_expectation,
    run_order_based_quepp,
)


def test_order_based_quepp_recovers_missing_high_order_signal_under_constant_rescaling():
    observable = PauliString.from_label("Z")
    rotation = PauliString.from_label("X")

    circuit = Circuit(1)
    circuit.add_pauli_rotation(rotation, math.pi / 4)
    circuit.add_pauli_rotation(rotation, math.pi / 5)

    paths = enumerate_cpt_paths(circuit, observable, max_order=0)
    truncated = cpt_estimate(paths)
    exact = exact_expectation(circuit, observable)

    backend = RescalingNoiseBackend(eta=0.8, mode="constant")
    result = run_order_based_quepp(
        circuit,
        observable,
        backend=backend,
        max_order=0,
        eta_strategy="median",
    )

    assert math.isclose(exact, math.cos((math.pi / 4) + (math.pi / 5)), rel_tol=1e-12)
    assert math.isclose(truncated, math.cos(math.pi / 4) * math.cos(math.pi / 5), rel_tol=1e-12)
    assert abs(truncated - exact) > 1e-2

    assert result.measured_path_count == 1
    assert math.isclose(result.eta, 0.8, rel_tol=1e-12)
    assert math.isclose(result.mitigated_value, exact, rel_tol=1e-12, abs_tol=1e-12)
