from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from .circuit import Circuit, CliffordGate, CliffordPauliRotation, PauliRotation
from .exact import exact_expectation
from .pauli import PauliString


class ExpectationBackend(Protocol):
    def estimate(self, circuit: Circuit, observable: PauliString) -> float:
        ...


@dataclass
class ExactStatevectorBackend:
    def estimate(self, circuit: Circuit, observable: PauliString) -> float:
        return exact_expectation(circuit, observable)


@dataclass
class RescalingNoiseBackend:
    eta: float = 0.9
    mode: str = "constant"
    shots: Optional[int] = None
    seed: Optional[int] = None
    ideal_estimator: Callable[[Circuit, PauliString], float] = exact_expectation
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def circuit_eta(self, circuit: Circuit) -> float:
        if self.mode == "constant":
            return self.eta
        if self.mode == "per_gate":
            return self.eta ** len(circuit.gates)
        raise ValueError(f"Unsupported rescaling mode: {self.mode}")

    def estimate(self, circuit: Circuit, observable: PauliString) -> float:
        ideal = self.ideal_estimator(circuit, observable)
        mean = float(np.clip(self.circuit_eta(circuit) * ideal, -1.0, 1.0))
        if self.shots is None:
            return mean

        if self.shots <= 0:
            raise ValueError("shots must be a positive integer.")
        probability = (mean + 1.0) / 2.0
        samples = self._rng.binomial(self.shots, probability)
        return float((2.0 * samples / self.shots) - 1.0)


@dataclass
class DensityMatrixNoiseBackend:
    single_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    amplitude_damping: float = 0.0
    shots: Optional[int] = None
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def estimate(self, circuit: Circuit, observable: PauliString) -> float:
        (
            DensityMatrix,
            Kraus,
            Operator,
            Pauli,
            QuantumCircuit,
            SparsePauliOp,
        ) = _load_qiskit_noise_primitives()

        if observable.n_qubits != circuit.n_qubits:
            raise ValueError("Observable size does not match the circuit size.")

        prep = QuantumCircuit(circuit.n_qubits)
        for qubit in range(circuit.n_qubits):
            if (circuit.initial_bitstring >> qubit) & 1:
                prep.x(qubit)
        rho = DensityMatrix.from_instruction(prep)

        single_depol = _single_qubit_depolarizing_channel(Kraus, self.single_qubit_depolarizing)
        two_depol = _two_qubit_depolarizing_channel(Kraus, self.two_qubit_depolarizing)
        amp_damp = _amplitude_damping_channel(Kraus, self.amplitude_damping)

        for gate in circuit.gates:
            rho = _evolve_ideal_gate(rho, gate, circuit.n_qubits, Operator, Pauli, QuantumCircuit)
            rho = _apply_noise_channels(rho, gate, single_depol, two_depol, amp_damp)

        observable_op = SparsePauliOp.from_list([(_to_qiskit_label(observable), 1.0)])
        mean = float(np.real_if_close(rho.expectation_value(observable_op)))

        if self.shots is None:
            return mean
        if self.shots <= 0:
            raise ValueError("shots must be a positive integer.")

        probability = float(np.clip((mean + 1.0) / 2.0, 0.0, 1.0))
        samples = self._rng.binomial(self.shots, probability)
        return float((2.0 * samples / self.shots) - 1.0)


def _load_qiskit_noise_primitives():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using Qiskit with Python 3.9 is deprecated.*")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Pauli, SparsePauliOp

    return DensityMatrix, Kraus, Operator, Pauli, QuantumCircuit, SparsePauliOp


def _to_qiskit_label(pauli: PauliString) -> str:
    return pauli.to_label()[::-1]


def _support_qubits(pauli: PauliString) -> list[int]:
    return [qubit for qubit in range(pauli.n_qubits) if pauli.local_char(qubit) != "I"]


def _single_qubit_depolarizing_channel(Kraus, probability: float):
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        raise ValueError("single_qubit_depolarizing must be smaller than 1.")

    identity = np.eye(2, dtype=complex)
    paulis = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    ops = [math.sqrt(1.0 - probability) * identity]
    ops.extend(math.sqrt(probability / 3.0) * pauli for pauli in paulis)
    return Kraus(ops)


def _two_qubit_depolarizing_channel(Kraus, probability: float):
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        raise ValueError("two_qubit_depolarizing must be smaller than 1.")

    single = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    ops = [math.sqrt(1.0 - probability) * np.eye(4, dtype=complex)]
    for left in "IXYZ":
        for right in "IXYZ":
            if left == "I" and right == "I":
                continue
            ops.append(math.sqrt(probability / 15.0) * np.kron(single[left], single[right]))
    return Kraus(ops)


def _amplitude_damping_channel(Kraus, gamma: float):
    if gamma <= 0.0:
        return None
    if gamma >= 1.0:
        raise ValueError("amplitude_damping must be smaller than 1.")

    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return Kraus([k0, k1])


def _evolve_ideal_gate(rho, gate, n_qubits: int, Operator, Pauli, QuantumCircuit):
    if isinstance(gate, CliffordGate):
        qc = QuantumCircuit(len(gate.qubits))
        if gate.symbol == "H":
            qc.h(0)
        elif gate.symbol == "S":
            qc.s(0)
        elif gate.symbol == "SDG":
            qc.sdg(0)
        elif gate.symbol == "X":
            qc.x(0)
        elif gate.symbol == "Y":
            qc.y(0)
        elif gate.symbol == "Z":
            qc.z(0)
        elif gate.symbol == "CX":
            qc.cx(0, 1)
        elif gate.symbol == "CZ":
            qc.cz(0, 1)
        else:
            raise ValueError(f"Unsupported Clifford gate: {gate.symbol}")
        return rho.evolve(Operator(qc), qargs=list(gate.qubits))

    if isinstance(gate, (PauliRotation, CliffordPauliRotation)):
        angle = gate.angle if isinstance(gate, PauliRotation) else (math.pi / 2.0)
        pauli_matrix = Pauli(_to_qiskit_label(gate.generator)).to_matrix()
        unitary = math.cos(angle / 2.0) * np.eye(2**n_qubits, dtype=complex) - 1j * math.sin(
            angle / 2.0
        ) * pauli_matrix
        return rho.evolve(Operator(unitary))

    raise TypeError(f"Unsupported gate type: {type(gate)!r}")


def _apply_noise_channels(rho, gate, single_depol, two_depol, amp_damp):
    qubits = ()
    if isinstance(gate, CliffordGate):
        qubits = gate.qubits
    elif isinstance(gate, (PauliRotation, CliffordPauliRotation)):
        qubits = tuple(_support_qubits(gate.generator))

    if len(qubits) == 1:
        if single_depol is not None:
            rho = rho.evolve(single_depol, qargs=list(qubits))
        if amp_damp is not None:
            rho = rho.evolve(amp_damp, qargs=list(qubits))
        return rho

    if len(qubits) == 2:
        if two_depol is not None:
            rho = rho.evolve(two_depol, qargs=list(qubits))
        elif single_depol is not None:
            for qubit in qubits:
                rho = rho.evolve(single_depol, qargs=[qubit])
        if amp_damp is not None:
            for qubit in qubits:
                rho = rho.evolve(amp_damp, qargs=[qubit])
        return rho

    for qubit in qubits:
        if single_depol is not None:
            rho = rho.evolve(single_depol, qargs=[qubit])
        if amp_damp is not None:
            rho = rho.evolve(amp_damp, qargs=[qubit])
    return rho
