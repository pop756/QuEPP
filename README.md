# QuEPP

Local research implementation of Quantum Enhanced Pauli Propagation (QuEPP).

This repository currently provides a clean Python v1 focused on:

- order-based CPT path enumeration
- Clifford-path expectation evaluation
- order-based QuEPP estimation with a pluggable expectation backend
- a synthetic rescaling-noise backend for local validation
- a density-matrix noisy backend that estimates pathwise `eta_i` from gate-level noise

Current scope is intentionally narrow so the core math is easy to inspect and test:

- initial state is a computational basis state, default `|0...0>`
- target circuits are built from Clifford gates plus Pauli rotations
- local exact validation uses Qiskit statevector simulation
- hardware/runtime integration is not included yet

## Supported gates

- Clifford: `H`, `S`, `SDG`, `X`, `Y`, `Z`, `CX`, `CZ`
- Non-Clifford: arbitrary Pauli rotations `exp(-i theta P / 2)` via `PauliRotation`

## Quick Start

```python
import math

from quepp import (
    Circuit,
    DensityMatrixNoiseBackend,
    PauliString,
    run_order_based_quepp,
)

observable = PauliString.from_label("Z")
rotation = PauliString.from_label("X")

circuit = Circuit(1)
circuit.add_pauli_rotation(rotation, math.pi / 4)
circuit.add_pauli_rotation(rotation, math.pi / 5)

backend = DensityMatrixNoiseBackend(
    single_qubit_depolarizing=0.03,
    amplitude_damping=0.01,
)
result = run_order_based_quepp(circuit, observable, backend=backend, max_order=0)

print(result.mitigated_value)
```

## Development

Run the test suite:

```bash
python -m pytest
```

Run the example:

```bash
python examples/basic_order_based_demo.py
python examples/noisy_backend_eta_experiment.py
python examples/five_qubit_noisy_experiment.py
python examples/five_qubit_ladder_experiment.py
```
