import math

from quepp import (
    Circuit,
    DensityMatrixNoiseBackend,
    PauliString,
    cpt_estimate,
    enumerate_cpt_paths,
    exact_expectation,
    run_order_based_quepp,
)


def test_density_matrix_backend_matches_exact_when_noise_is_zero():
    observable = PauliString.from_label("Z")
    circuit = Circuit(1)
    circuit.add_clifford("H", 0)
    circuit.add_pauli_rotation(PauliString.from_label("X"), math.pi / 4)

    backend = DensityMatrixNoiseBackend()

    assert math.isclose(
        backend.estimate(circuit, observable),
        exact_expectation(circuit, observable),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_quepp_estimates_eta_from_noisy_backend_and_improves_over_truncated_cpt():
    observable = PauliString.from_label("Z")
    rotation = PauliString.from_label("X")

    circuit = Circuit(1)
    circuit.add_clifford("S", 0)
    circuit.add_pauli_rotation(rotation, math.pi / 4)
    circuit.add_pauli_rotation(rotation, math.pi / 5)

    backend = DensityMatrixNoiseBackend(single_qubit_depolarizing=0.03, amplitude_damping=0.01)
    result = run_order_based_quepp(circuit, observable, backend=backend, max_order=0, eta_strategy="median")

    exact = exact_expectation(circuit, observable)
    truncated = cpt_estimate(enumerate_cpt_paths(circuit, observable, max_order=0))
    noisy = backend.estimate(circuit, observable)

    assert result.measured_path_count == 1
    assert 0.0 < result.eta < 1.0
    assert abs(result.mitigated_value - exact) < abs(truncated - exact)
    assert abs(result.mitigated_value - exact) < abs(noisy - exact)
