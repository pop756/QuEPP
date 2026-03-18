from __future__ import annotations

from typing import Dict, Tuple

from .circuit import Circuit, CliffordGate, CliffordPauliRotation
from .pauli import PauliString


_SINGLE_QUBIT_MAPS: Dict[str, Tuple[Tuple[Tuple[int, int], int], ...]] = {
    "H": (
        ((0, 0), 1),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), -1),
    ),
    "S": (
        ((0, 0), 1),
        ((1, 1), -1),
        ((0, 1), 1),
        ((1, 0), 1),
    ),
    "SDG": (
        ((0, 0), 1),
        ((1, 1), 1),
        ((0, 1), 1),
        ((1, 0), -1),
    ),
    "X": (
        ((0, 0), 1),
        ((1, 0), 1),
        ((0, 1), -1),
        ((1, 1), -1),
    ),
    "Y": (
        ((0, 0), 1),
        ((1, 0), -1),
        ((0, 1), -1),
        ((1, 1), 1),
    ),
    "Z": (
        ((0, 0), 1),
        ((1, 0), -1),
        ((0, 1), 1),
        ((1, 1), -1),
    ),
}

_TWO_QUBIT_MAPS: Dict[str, Dict[str, tuple[str, int]]] = {
    "CX": {
        "II": ("II", 1),
        "XI": ("XX", 1),
        "YI": ("YX", 1),
        "ZI": ("ZI", 1),
        "IX": ("IX", 1),
        "IY": ("ZY", 1),
        "IZ": ("ZZ", 1),
        "XX": ("XI", 1),
        "XY": ("YZ", 1),
        "XZ": ("YY", -1),
        "YX": ("YI", 1),
        "YY": ("XZ", -1),
        "YZ": ("XY", 1),
        "ZX": ("ZX", 1),
        "ZY": ("IY", 1),
        "ZZ": ("IZ", 1),
    },
    "CZ": {
        "II": ("II", 1),
        "XI": ("XZ", 1),
        "YI": ("YZ", 1),
        "ZI": ("ZI", 1),
        "IX": ("ZX", 1),
        "IY": ("ZY", 1),
        "IZ": ("IZ", 1),
        "XX": ("YY", 1),
        "XY": ("YX", -1),
        "XZ": ("XI", 1),
        "YX": ("XY", -1),
        "YY": ("XX", 1),
        "YZ": ("YI", 1),
        "ZX": ("IX", 1),
        "ZY": ("IY", 1),
        "ZZ": ("ZZ", 1),
    },
}


def _local_index(pauli: PauliString, qubit: int) -> int:
    x_bit = 1 if (pauli.x_mask >> qubit) & 1 else 0
    z_bit = 1 if (pauli.z_mask >> qubit) & 1 else 0
    return x_bit + 2 * z_bit


def apply_clifford_gate(pauli: PauliString, gate: CliffordGate) -> tuple[int, PauliString]:
    if gate.symbol in _SINGLE_QUBIT_MAPS:
        if len(gate.qubits) != 1:
            raise ValueError(f"{gate.symbol} expects a single qubit.")
        qubit = gate.qubits[0]
        (x_bit, z_bit), sign = _SINGLE_QUBIT_MAPS[gate.symbol][_local_index(pauli, qubit)]
        bit = 1 << qubit
        x_mask = (pauli.x_mask & ~bit) | (x_bit << qubit)
        z_mask = (pauli.z_mask & ~bit) | (z_bit << qubit)
        return sign, PauliString(n_qubits=pauli.n_qubits, x_mask=x_mask, z_mask=z_mask)

    if gate.symbol in _TWO_QUBIT_MAPS:
        if len(gate.qubits) != 2:
            raise ValueError(f"{gate.symbol} expects two qubits.")
        qubit0, qubit1 = gate.qubits
        key = pauli.local_char(qubit0) + pauli.local_char(qubit1)
        out_label, sign = _TWO_QUBIT_MAPS[gate.symbol][key]
        out = pauli.with_local_char(qubit0, out_label[0]).with_local_char(qubit1, out_label[1])
        return sign, out

    raise ValueError(f"Unsupported Clifford gate: {gate.symbol}")


def apply_clifford_pauli_rotation(
    pauli: PauliString,
    gate: CliffordPauliRotation,
) -> tuple[int, PauliString]:
    if pauli.commutes_with(gate.generator):
        return 1, pauli

    phase, product = gate.generator.multiply(pauli)
    if phase == 1:
        return -1, product
    if phase == 3:
        return 1, product
    raise ValueError("Anticommuting Pauli strings must produce an imaginary phase.")


def clifford_expectation(circuit: Circuit, observable: PauliString) -> float:
    if observable.n_qubits != circuit.n_qubits:
        raise ValueError("Observable size does not match the circuit size.")

    sign = 1
    current = observable
    for gate in reversed(circuit.gates):
        if isinstance(gate, CliffordGate):
            gate_sign, current = apply_clifford_gate(current, gate)
        elif isinstance(gate, CliffordPauliRotation):
            gate_sign, current = apply_clifford_pauli_rotation(current, gate)
        else:
            raise TypeError("clifford_expectation only supports Clifford circuits.")
        sign *= gate_sign

    return float(sign * current.expectation_on_basis(circuit.initial_bitstring))
