from __future__ import annotations

import math
import warnings

import numpy as np

from .circuit import Circuit, CliffordGate, CliffordPauliRotation, PauliRotation
from .pauli import PauliString


def _to_qiskit_label(pauli: PauliString) -> str:
    return pauli.to_label()[::-1]


def exact_expectation(circuit: Circuit, observable: PauliString) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using Qiskit with Python 3.9 is deprecated.*")
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit.quantum_info import SparsePauliOp, Statevector
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("qiskit is required for exact expectation evaluation.") from exc

    if observable.n_qubits != circuit.n_qubits:
        raise ValueError("Observable size does not match the circuit size.")

    quantum_circuit = QuantumCircuit(circuit.n_qubits)
    for qubit in range(circuit.n_qubits):
        if (circuit.initial_bitstring >> qubit) & 1:
            quantum_circuit.x(qubit)

    for gate in circuit.gates:
        if isinstance(gate, CliffordGate):
            if gate.symbol == "H":
                quantum_circuit.h(gate.qubits[0])
            elif gate.symbol == "S":
                quantum_circuit.s(gate.qubits[0])
            elif gate.symbol == "SDG":
                quantum_circuit.sdg(gate.qubits[0])
            elif gate.symbol == "X":
                quantum_circuit.x(gate.qubits[0])
            elif gate.symbol == "Y":
                quantum_circuit.y(gate.qubits[0])
            elif gate.symbol == "Z":
                quantum_circuit.z(gate.qubits[0])
            elif gate.symbol == "CX":
                quantum_circuit.cx(*gate.qubits)
            elif gate.symbol == "CZ":
                quantum_circuit.cz(*gate.qubits)
            else:
                raise ValueError(f"Unsupported Clifford gate: {gate.symbol}")
            continue

        if isinstance(gate, (PauliRotation, CliffordPauliRotation)):
            angle = gate.angle if isinstance(gate, PauliRotation) else (math.pi / 2.0)
            sparse_pauli = SparsePauliOp.from_list([(_to_qiskit_label(gate.generator), 1.0)])
            quantum_circuit.append(
                PauliEvolutionGate(sparse_pauli, time=angle / 2.0),
                tuple(range(circuit.n_qubits)),
            )
            continue

        raise TypeError(f"Unsupported gate type: {type(gate)!r}")

    with warnings.catch_warnings():
        try:
            from scipy.sparse import SparseEfficiencyWarning

            warnings.simplefilter("ignore", SparseEfficiencyWarning)
        except ImportError:  # pragma: no cover - optional SciPy warning class
            pass
        state = Statevector.from_instruction(quantum_circuit)
    observable_op = SparsePauliOp.from_list([(_to_qiskit_label(observable), 1.0)])
    value = state.expectation_value(observable_op)
    return float(np.real_if_close(value))
