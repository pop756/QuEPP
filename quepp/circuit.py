from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .pauli import PauliString


@dataclass(frozen=True)
class Gate:
    pass


@dataclass(frozen=True)
class CliffordGate(Gate):
    symbol: str
    qubits: Tuple[int, ...]

    def __post_init__(self) -> None:
        symbol = self.symbol.upper()
        if symbol == "CNOT":
            symbol = "CX"
        if symbol == "SDAG":
            symbol = "SDG"
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "qubits", tuple(self.qubits))


@dataclass(frozen=True)
class PauliRotation(Gate):
    generator: PauliString
    angle: float


@dataclass(frozen=True)
class CliffordPauliRotation(Gate):
    generator: PauliString


@dataclass
class Circuit:
    n_qubits: int
    gates: List[Gate] = field(default_factory=list)
    initial_bitstring: int = 0

    def append(self, gate: Gate) -> "Circuit":
        if isinstance(gate, (PauliRotation, CliffordPauliRotation)):
            if gate.generator.n_qubits != self.n_qubits:
                raise ValueError("Gate generator size does not match the circuit size.")
        for qubit in getattr(gate, "qubits", ()):
            if qubit < 0 or qubit >= self.n_qubits:
                raise ValueError("Gate qubit index out of range.")
        self.gates.append(gate)
        return self

    def extend(self, gates: Iterable[Gate]) -> "Circuit":
        for gate in gates:
            self.append(gate)
        return self

    def add_clifford(self, symbol: str, *qubits: int) -> "Circuit":
        return self.append(CliffordGate(symbol=symbol, qubits=tuple(qubits)))

    def add_pauli_rotation(self, generator: PauliString, angle: float) -> "Circuit":
        return self.append(PauliRotation(generator=generator, angle=angle))

    def copy(self) -> "Circuit":
        return Circuit(
            n_qubits=self.n_qubits,
            gates=list(self.gates),
            initial_bitstring=self.initial_bitstring,
        )

    def signature(self) -> tuple:
        entries = []
        for gate in self.gates:
            if isinstance(gate, CliffordGate):
                entries.append(("clifford", gate.symbol, gate.qubits))
            elif isinstance(gate, PauliRotation):
                entries.append(
                    (
                        "rotation",
                        gate.generator.x_mask,
                        gate.generator.z_mask,
                        round(gate.angle, 14),
                    )
                )
            elif isinstance(gate, CliffordPauliRotation):
                entries.append(("clifford_rotation", gate.generator.x_mask, gate.generator.z_mask))
            else:
                raise TypeError(f"Unsupported gate type: {type(gate)!r}")
        return (self.n_qubits, self.initial_bitstring, tuple(entries))

    @classmethod
    def from_gates(
        cls,
        n_qubits: int,
        gates: Sequence[Gate],
        initial_bitstring: int = 0,
    ) -> "Circuit":
        circuit = cls(n_qubits=n_qubits, initial_bitstring=initial_bitstring)
        circuit.extend(gates)
        return circuit
