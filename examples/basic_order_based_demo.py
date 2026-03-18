import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quepp import (
    Circuit,
    PauliString,
    RescalingNoiseBackend,
    cpt_estimate,
    enumerate_cpt_paths,
    exact_expectation,
    run_order_based_quepp,
)


def main() -> None:
    observable = PauliString.from_label("Z")
    rotation = PauliString.from_label("X")

    circuit = Circuit(1)
    circuit.add_pauli_rotation(rotation, math.pi / 4)
    circuit.add_pauli_rotation(rotation, math.pi / 8)

    paths = enumerate_cpt_paths(circuit, observable, max_order=0)
    print(paths)
    backend = RescalingNoiseBackend(eta=0.8, mode="constant")
    result = run_order_based_quepp(circuit, observable, backend=backend, max_order=0)

    print("Exact target expectation :", round(exact_expectation(circuit, observable), 6))
    print("Truncated CPT estimate   :", round(cpt_estimate(paths), 6))
    print("Estimated eta            :", round(result.eta, 6))
    print("Noisy value :", round(result.residual_noisy_value,6))
    print("ideal value :", round(result.ideal_cpt_value,6))
    print("QuEPP mitigated value    :", round(result.mitigated_value, 6))


if __name__ == "__main__":
    main()
