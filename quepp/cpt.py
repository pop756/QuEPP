from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .circuit import Circuit, CliffordGate, CliffordPauliRotation, PauliRotation
from .pauli import PauliString
from .propagation import apply_clifford_gate, apply_clifford_pauli_rotation


@dataclass(frozen=True)
class CPTPath:
    order: int
    weight: float
    clifford_circuit: Circuit
    initial_pauli: PauliString
    clifford_sign: int
    ideal_expectation: float
    branch_trace: Tuple[str, ...]


def enumerate_cpt_paths(
    circuit: Circuit,
    observable: PauliString,
    max_order: int,
    weight_cutoff: float = 1e-15,
) -> list[CPTPath]:
    if observable.n_qubits != circuit.n_qubits:
        raise ValueError("Observable size does not match the circuit size.")

    paths: List[CPTPath] = []
    reversed_gates = list(reversed(circuit.gates))

    def dfs(
        gate_index: int,
        current_pauli: PauliString,
        clifford_sign: int,
        weight: float,
        order: int,
        clifford_gates_reversed: Tuple[object, ...],
        branch_trace: Tuple[str, ...],
    ) -> None:
        if order > max_order or abs(weight) <= weight_cutoff:
            return

        if gate_index == len(reversed_gates):
            clifford_circuit = Circuit.from_gates(
                n_qubits=circuit.n_qubits,
                gates=list(reversed(clifford_gates_reversed)),
                initial_bitstring=circuit.initial_bitstring,
            )
            ideal = float(clifford_sign * current_pauli.expectation_on_basis(circuit.initial_bitstring))
            paths.append(
                CPTPath(
                    order=order,
                    weight=weight,
                    clifford_circuit=clifford_circuit,
                    initial_pauli=current_pauli,
                    clifford_sign=clifford_sign,
                    ideal_expectation=ideal,
                    branch_trace=branch_trace,
                )
            )
            return

        gate = reversed_gates[gate_index]
        if isinstance(gate, CliffordGate):
            gate_sign, next_pauli = apply_clifford_gate(current_pauli, gate)
            dfs(
                gate_index + 1,
                next_pauli,
                clifford_sign * gate_sign,
                weight,
                order,
                clifford_gates_reversed + (gate,),
                branch_trace + (f"{gate.symbol}",),
            )
            return

        if isinstance(gate, PauliRotation):
            if current_pauli.commutes_with(gate.generator):
                dfs(
                    gate_index + 1,
                    current_pauli,
                    clifford_sign,
                    weight,
                    order,
                    clifford_gates_reversed,
                    branch_trace + ("commute",),
                )
                return

            cos_weight = weight * math.cos(gate.angle)
            dfs(
                gate_index + 1,
                current_pauli,
                clifford_sign,
                cos_weight,
                order,
                clifford_gates_reversed,
                branch_trace + ("cos",),
            )

            rotation_gate = CliffordPauliRotation(generator=gate.generator)
            sine_sign, next_pauli = apply_clifford_pauli_rotation(current_pauli, rotation_gate)
            sin_weight = weight * math.sin(gate.angle)
            dfs(
                gate_index + 1,
                next_pauli,
                clifford_sign * sine_sign,
                sin_weight,
                order + 1,
                clifford_gates_reversed + (rotation_gate,),
                branch_trace + ("sin",),
            )
            return

        raise TypeError(f"Unsupported gate in target circuit: {type(gate)!r}")

    dfs(
        gate_index=0,
        current_pauli=observable,
        clifford_sign=1,
        weight=1.0,
        order=0,
        clifford_gates_reversed=tuple(),
        branch_trace=tuple(),
    )
    return paths


def cpt_estimate(paths: Sequence[CPTPath]) -> float:
    return float(sum(path.weight * path.ideal_expectation for path in paths))

