import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quepp import (
    Circuit,
    DensityMatrixNoiseBackend,
    PauliString,
    cpt_estimate,
    enumerate_cpt_paths,
    exact_expectation,
    run_order_based_quepp,
)


def build_five_qubit_ladder_experiment() -> tuple[Circuit, PauliString, int]:
    circuit = Circuit(5)
    for _ in range(5):
        circuit.add_pauli_rotation(PauliString.from_sparse(5, {0: "X"}), math.pi / 4)
        circuit.add_pauli_rotation(PauliString.from_sparse(5, {1: "X"}), math.pi / 4)
        circuit.add_pauli_rotation(PauliString.from_sparse(5, {2: "Y"}), math.pi / 4)
        circuit.add_pauli_rotation(PauliString.from_sparse(5, {3: "X"}), math.pi / 4)
        circuit.add_pauli_rotation(PauliString.from_sparse(5, {4: "X"}), math.pi / 4)
        for qubit in range(4):
            circuit.add_clifford("CX", qubit, qubit + 1)

    observable = PauliString.from_label("IIIIZ")
    max_order = 5
    return circuit, observable, max_order


def main() -> None:
    circuit, observable, max_order = build_five_qubit_ladder_experiment()

    backend = DensityMatrixNoiseBackend(
        single_qubit_depolarizing=0.005,
        two_qubit_depolarizing=0.01,
        amplitude_damping=0.00,
    )

    paths = enumerate_cpt_paths(circuit, observable, max_order=max_order)
    measured_path_count = sum(1 for path in paths if path.ideal_expectation != 0.0)
    if measured_path_count == 0:
        raise RuntimeError(
            "No non-zero Clifford paths were found for eta estimation in this ladder example. "
            "Choose a different observable or max_order."
        )
    result = run_order_based_quepp(circuit, observable, backend=backend, max_order=max_order)

    exact = exact_expectation(circuit, observable)
    noisy = backend.estimate(circuit, observable)
    truncated = cpt_estimate(paths)

    print("Observable                 :", observable.to_label())
    print("Number of qubits           :", circuit.n_qubits)
    print("Number of gates            :", len(circuit.gates))
    print("Max order                  :", max_order)
    print("Path count up to max_order :", result.path_count)
    print("Measured path count        :", result.measured_path_count)
    print("Exact target expectation   :", round(exact, 6))
    print("Noisy target expectation   :", round(noisy, 6))
    print("Truncated CPT estimate     :", round(truncated, 6))
    print("Estimated eta values       :", [round(value, 6) for value in result.eta_values])
    print("Global eta                 :", round(result.eta, 6))
    print("QuEPP mitigated expectation:", round(result.mitigated_value, 6))
    print("Abs error noisy            :", round(abs(noisy - exact), 6))
    print("Abs error truncated CPT    :", round(abs(truncated - exact), 6))
    print("Abs error QuEPP            :", round(abs(result.mitigated_value - exact), 6))


if __name__ == "__main__":
    main()
