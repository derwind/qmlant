from __future__ import annotations

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit

from qmlant.neural_networks.utils import Identity, Pauli, PauliZ
from qmlant.neural_networks.utils import (
    circuit_to_einsum_expectation as nnu_circuit_to_einsum_expectation,
)


class IsingConverter:
    def __init__(self, qubo: dict[tuple[str, str], float]):
        self._qubo = qubo
        self.num_qubits = _calc_num_qubits(qubo)

    def get_ising(self) -> tuple[dict[tuple[str] | tuple[str, str], float], float]:
        ising_dict: dict[tuple[str] | tuple[str, str], float] = {}
        offset = 0.0

        for k, v in self._qubo.items():
            left, right = k
            ln = int(left[1:])
            rn = int(right[1:])
            new_k: tuple[str] | tuple[str, str]
            if rn < ln:
                ln, rn = rn, ln
            if ln == rn:
                new_k = (f"z{ln}",)
                ising_dict.setdefault(new_k, 0.0)
                ising_dict[new_k] += -v / 2
                offset += 1 / 2
            else:
                new_k = (f"z{ln}", f"z{rn}")
                ising_dict.setdefault(new_k, 0.0)
                ising_dict[new_k] += v / 4
                new_k = (f"z{ln}",)
                ising_dict.setdefault(new_k, 0.0)
                ising_dict[new_k] += -v / 4
                new_k = (f"z{rn}",)
                ising_dict.setdefault(new_k, 0.0)
                ising_dict[new_k] += -v / 4
                offset += 1 / 4

        ising_dict = dict(sorted(ising_dict.items(), key=lambda k_v: self._calc_key(k_v[0])))
        return ising_dict, offset

    def _calc_key(self, k: tuple[str] | tuple[str, str]) -> int:
        if len(k) == 1:
            left = k[0]
            ln = int(left[1:])
            return self.num_qubits * ln - 1
        elif len(k) == 2:
            left, right = k  # type: ignore
            ln = int(left[1:])
            rn = int(right[1:])
            return self.num_qubits * self.num_qubits * ln + self.num_qubits * rn
        else:
            raise ValueError(f"len(k) = {len(k)} must be one or two.")


class HamiltonianConverter:
    def __init__(self, ising_dict: dict[tuple[str] | tuple[str, str], float]):
        self._ising_dict = ising_dict
        self.num_qubits = _calc_num_qubits(ising_dict)

    def get_hamiltonian(self) -> tuple[list[cp.ndarray], np.ndarray]:
        """get Hamiltonian array and coefficients for `a_{12} Z_1 Z_2 + a_{34} Z_3 Z_4` etc.

        Returns:
            tuple[list[cp.array]: Hamiltonian array, e.g., Z_1 Z_2 + Z_3 Z_4
            np.ndarray: Hamiltonian coefficients, e.g. [a_{12}, a_{34}]
        """

        I = Identity(xp=cp)  # noqa: E741
        Z = PauliZ(xp=cp)

        hamiltonian: list[cp.ndarray] = []
        for i in range(self.num_qubits):
            row = []
            for k in self._ising_dict.keys():
                if len(k) == 1:
                    left = k[0]
                    ln = int(left[1:])
                    rn = None
                elif len(k) == 2:
                    left, right = k  # type: ignore
                    ln = int(left[1:])
                    rn = int(right[1:])
                else:
                    raise ValueError(f"len(k) = {len(k)} must be one or two.")

                if ln == i or rn == i:
                    row.append(Z)
                else:
                    row.append(I)
            hamiltonian.append(cp.array(row))

        return hamiltonian, np.array(list(self._ising_dict.values()))


def circuit_to_einsum_expectation(
    qc_pl: QuantumCircuit, hamiltonian: list[cp.ndarray], coefficients: np.ndarray | None = None
) -> tuple[str, list[cp.ndarray], dict[str, tuple[list[int], list[int], Pauli]]]:
    dummy_hamiltonian = "Z" * qc_pl.num_qubits
    expr, operands, pname2locs = nnu_circuit_to_einsum_expectation(qc_pl, dummy_hamiltonian)
    hamiltonian_locs = _find_dummy_hamiltonian(operands)
    max_hamiltonian_loc = max(hamiltonian_locs)
    es = expr.split("->")[0].split(",")
    for loc in hamiltonian_locs:
        es[loc] = "食" + es[loc]  # hamu
    if coefficients is not None:
        es.insert(max_hamiltonian_loc, "食")
    expr = ",".join(es) + "->"
    new_pname2locs: dict[str, tuple[list[int], list[int], Pauli]] = {}
    # shift +1 for dag_locs due to embedding of coefficients into TN
    for name, (locs, dag_locs, make_paulis) in pname2locs.items():
        shifted_dag_locs = [v + 1 for v in dag_locs]
        new_pname2locs[name] = (locs, shifted_dag_locs, make_paulis)

    for ham, locs in zip(hamiltonian, hamiltonian_locs):  # type: ignore
        operands[locs] = ham  # type: ignore
    if coefficients is not None:
        operands.insert(max_hamiltonian_loc, cp.array(coefficients, dtype=complex))

    operands = _convert_dtype(operands, np.complex64)  # for better performance

    return expr, operands, new_pname2locs


def _find_dummy_hamiltonian(operands: list[cp.ndarray]) -> list[int]:
    Z = PauliZ(xp=cp)

    locs = []
    for i, op in enumerate(operands):
        if cp.all(op == Z):
            locs.append(i)
    return locs


def _convert_dtype(operands, dtype):
    return [op.astype(dtype) for op in operands]


def _calc_num_qubits(
    qubo_ising: dict[tuple[str, str], float] | dict[tuple[str] | tuple[str, str], float]
) -> int:
    numbers = set()
    for k in qubo_ising:
        if len(k) == 1:
            left = k[0]
            ln = int(left[1:])
            numbers.add(ln)
        elif len(k) == 2:
            left, right = k  # type: ignore
            ln = int(left[1:])
            rn = int(right[1:])
            numbers.add(ln)
            numbers.add(rn)
        else:
            raise ValueError(f"len(k) = {len(k)} must be one or two.")
    return max(numbers) + 1
