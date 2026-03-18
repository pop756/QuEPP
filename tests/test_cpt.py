import math

from quepp import Circuit, PauliString, cpt_estimate, enumerate_cpt_paths


def test_single_rx_cpt_matches_analytic_expansion():
    observable = PauliString.from_label("Z")
    circuit = Circuit(1)
    circuit.add_pauli_rotation(PauliString.from_label("X"), math.pi / 4)

    paths = enumerate_cpt_paths(circuit, observable, max_order=1)

    assert len(paths) == 2

    order_zero = next(path for path in paths if path.order == 0)
    order_one = next(path for path in paths if path.order == 1)

    assert math.isclose(order_zero.weight, math.cos(math.pi / 4), rel_tol=1e-12)
    assert order_zero.ideal_expectation == 1.0
    assert order_zero.initial_pauli.to_label() == "Z"

    assert math.isclose(order_one.weight, math.sin(math.pi / 4), rel_tol=1e-12)
    assert order_one.ideal_expectation == 0.0
    assert order_one.initial_pauli.to_label() == "Y"

    assert math.isclose(cpt_estimate(paths), math.cos(math.pi / 4), rel_tol=1e-12)

