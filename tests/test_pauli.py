import math

from quepp import PauliString


def test_pauli_multiplication_tracks_phase():
    x = PauliString.from_label("X")
    z = PauliString.from_label("Z")

    phase, out = x.multiply(z)

    assert phase == 3
    assert out.to_label() == "Y"


def test_pauli_expectation_on_basis_state():
    pauli = PauliString.from_label("ZZI")

    assert pauli.expectation_on_basis(0b000) == 1
    assert pauli.expectation_on_basis(0b001) == -1
    assert pauli.expectation_on_basis(0b011) == 1

