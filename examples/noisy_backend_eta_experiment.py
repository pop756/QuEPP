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


def main() -> None:
    observable = PauliString.from_label("Z")
    rotation = PauliString.from_label("X")

    circuit = Circuit(1)
    circuit.add_pauli_rotation(rotation, math.pi / 8)
    circuit.add_clifford("S", 0)

    backend = DensityMatrixNoiseBackend(
        single_qubit_depolarizing=0.05,
        amplitude_damping=0.05,
    )

    max_order = 0
    paths = enumerate_cpt_paths(circuit, observable, max_order=max_order)
    result = run_order_based_quepp(circuit, observable, backend=backend, max_order=max_order)

    print("Circuit gates              :", circuit.gates)
    print("Exact target expectation   :", round(exact_expectation(circuit, observable), 6))
    print("Noisy target expectation   :", round(backend.estimate(circuit, observable), 6))
    print("Truncated CPT estimate     :", round(cpt_estimate(paths), 6))
    print("Estimated eta values       :", [round(value, 6) for value in result.eta_values])
    print("Global eta                 :", round(result.eta, 6))
    print("QuEPP mitigated expectation:", round(result.mitigated_value, 6))


if __name__ == "__main__":
    main()
