from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_CHAR_TO_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

_BITS_TO_CHAR = {value: key for key, value in _CHAR_TO_BITS.items()}

_LOCAL_PRODUCT = {
    ("I", "I"): ("I", 0),
    ("I", "X"): ("X", 0),
    ("I", "Y"): ("Y", 0),
    ("I", "Z"): ("Z", 0),
    ("X", "I"): ("X", 0),
    ("Y", "I"): ("Y", 0),
    ("Z", "I"): ("Z", 0),
    ("X", "X"): ("I", 0),
    ("Y", "Y"): ("I", 0),
    ("Z", "Z"): ("I", 0),
    ("X", "Y"): ("Z", 1),
    ("Y", "X"): ("Z", 3),
    ("X", "Z"): ("Y", 3),
    ("Z", "X"): ("Y", 1),
    ("Y", "Z"): ("X", 1),
    ("Z", "Y"): ("X", 3),
}


def _popcount(value: int) -> int:
    return bin(int(value)).count("1")


@dataclass(frozen=True)
class PauliString:
    n_qubits: int
    x_mask: int = 0
    z_mask: int = 0

    @classmethod
    def identity(cls, n_qubits: int) -> "PauliString":
        return cls(n_qubits=n_qubits)

    @classmethod
    def from_label(cls, label: str) -> "PauliString":
        label = label.upper()
        x_mask = 0
        z_mask = 0
        for qubit, char in enumerate(label):
            x_bit, z_bit = _CHAR_TO_BITS[char]
            if x_bit:
                x_mask |= 1 << qubit
            if z_bit:
                z_mask |= 1 << qubit
        return cls(n_qubits=len(label), x_mask=x_mask, z_mask=z_mask)

    @classmethod
    def from_sparse(cls, n_qubits: int, entries: Mapping[int, str]) -> "PauliString":
        x_mask = 0
        z_mask = 0
        for qubit, char in entries.items():
            x_bit, z_bit = _CHAR_TO_BITS[char.upper()]
            if x_bit:
                x_mask |= 1 << qubit
            if z_bit:
                z_mask |= 1 << qubit
        return cls(n_qubits=n_qubits, x_mask=x_mask, z_mask=z_mask)

    def to_label(self) -> str:
        chars = []
        for qubit in range(self.n_qubits):
            x_bit = 1 if (self.x_mask >> qubit) & 1 else 0
            z_bit = 1 if (self.z_mask >> qubit) & 1 else 0
            chars.append(_BITS_TO_CHAR[(x_bit, z_bit)])
        return "".join(chars)

    def local_char(self, qubit: int) -> str:
        x_bit = 1 if (self.x_mask >> qubit) & 1 else 0
        z_bit = 1 if (self.z_mask >> qubit) & 1 else 0
        return _BITS_TO_CHAR[(x_bit, z_bit)]

    def with_local_char(self, qubit: int, char: str) -> "PauliString":
        x_bit, z_bit = _CHAR_TO_BITS[char.upper()]
        bit = 1 << qubit
        x_mask = (self.x_mask & ~bit) | (x_bit << qubit)
        z_mask = (self.z_mask & ~bit) | (z_bit << qubit)
        return PauliString(n_qubits=self.n_qubits, x_mask=x_mask, z_mask=z_mask)

    def weight(self) -> int:
        return _popcount(self.x_mask | self.z_mask)

    def commutes_with(self, other: "PauliString") -> bool:
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli strings must have the same number of qubits.")
        symplectic = _popcount((self.x_mask & other.z_mask) ^ (self.z_mask & other.x_mask))
        return symplectic % 2 == 0

    def multiply(self, other: "PauliString") -> tuple[int, "PauliString"]:
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli strings must have the same number of qubits.")

        phase = 0
        x_mask = 0
        z_mask = 0

        for qubit in range(self.n_qubits):
            lhs = self.local_char(qubit)
            rhs = other.local_char(qubit)
            out_char, out_phase = _LOCAL_PRODUCT[(lhs, rhs)]
            phase = (phase + out_phase) % 4
            x_bit, z_bit = _CHAR_TO_BITS[out_char]
            if x_bit:
                x_mask |= 1 << qubit
            if z_bit:
                z_mask |= 1 << qubit

        return phase, PauliString(n_qubits=self.n_qubits, x_mask=x_mask, z_mask=z_mask)

    def expectation_on_basis(self, bitstring: int = 0) -> int:
        if self.x_mask:
            return 0
        parity = _popcount(self.z_mask & bitstring) % 2
        return -1 if parity else 1

    def __str__(self) -> str:
        return self.to_label()
