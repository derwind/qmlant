from __future__ import annotations

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit

from qmlant.neural_networks.utils import (
    Identity,
    MatZero,
    ParameterName2Locs,
    Pauli,
    PauliLocs,
    PauliZ,
    SplittedOperandsDict,
)
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
                offset += v / 2
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
                offset += v / 4

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
    qc_pl: QuantumCircuit,
    hamiltonian: list[cp.ndarray],
    coefficients: np.ndarray | None = None,
    prioritize_performance: bool = True,
    n_partial_hamiltonian: int = 1,
) -> (
    tuple[str, list[cp.ndarray], ParameterName2Locs]
    | tuple[
        str,
        SplittedOperandsDict,
        ParameterName2Locs,
    ]
):
    """CircuitToEinsum with hamiltonian embedded for `expectation`

    Args:
        qc_pl (QuantumCircuit): placeholder `QuantumCircuit` with `ParameterVector`
        hamiltonian (list[cp.ndarray]): Hamiltonian list
        coefficients (np.ndarray | None): coefficients of Hamiltonian
        n_partial_hamiltonian (int): III+IIZ, IZI,ZII, ZIZ+ZZI for
            III + IIZ + IZI + ZII + ZIZ + ZZI if n_partial_hamiltonian = 2

    Returns:
        str: `expr`
        list[cp.ndarray] | tuple[list[int], tuple[tuple[cp.ndarray]], list[int]]: `operands`,
            or tuple of `operands`, list of partial hamitonian list and hamiltonian locs
        dict[str, tuple[list[int], list[int], Pauli]]: dict of parameter name to locs
    """

    # TN with dummy hamiltonian
    dummy_hamiltonian = "Z" * qc_pl.num_qubits
    expr, operands, pname2locs = nnu_circuit_to_einsum_expectation(qc_pl, dummy_hamiltonian)
    hamiltonian_locs = _find_dummy_hamiltonian(operands)
    coefficients_loc = max(hamiltonian_locs) + 1

    # update expr,
    es = expr.split("->")[0].split(",")
    for loc in hamiltonian_locs:
        es[loc] = "食" + es[loc]  # hamu
    # and embed coefficients if needed with updating pname2locs
    new_pname2locs: ParameterName2Locs = {}
    if coefficients is None:
        new_pname2locs = pname2locs
    else:
        es.insert(coefficients_loc, "食")
        # shift +1 for dag_locs due to embedding of coefficients into TN
        for name, pauli2locs in pname2locs.items():
            new_pauli2locs: dict[Pauli, PauliLocs] = {}
            for make_paulis, (locs, dag_locs) in pauli2locs.items():
                shifted_dag_locs = [v + 1 for v in dag_locs]
                new_pauli2locs[make_paulis] = PauliLocs(locs, shifted_dag_locs)
            new_pname2locs[name] = new_pauli2locs
        coefficients = cp.array(coefficients, dtype=complex)
        operands.insert(coefficients_loc, coefficients)
    new_expr = ",".join(es) + "->"

    # update operands with real hamiltonian
    if n_partial_hamiltonian <= 1:
        for ham, locs in zip(hamiltonian, hamiltonian_locs):  # type: ignore
            operands[locs] = ham  # type: ignore
        new_operands = _convert_dtype(operands, prioritize_performance)  # for better performance
    else:
        operands_ = _convert_dtype(operands, prioritize_performance)  # for better performance
        # split hamiltonian into partial Hamiltonians in order to prevent VRAM overflow
        length = n_partial_hamiltonian
        if len(hamiltonian[0]) % length != 0:
            n_deficiency = int(np.ceil(len(hamiltonian[0]) / length)) * length - len(hamiltonian[0])
            zero = MatZero()
            padding = cp.array([zero] * n_deficiency, dtype=complex)
            hamiltonian = [cp.concatenate([ham, padding], axis=0) for ham in hamiltonian]

            if coefficients is not None:
                padding = cp.array([0] * n_deficiency, dtype=complex)
                coefficients = cp.concatenate([coefficients, padding], axis=0)

        hamiltonian = _convert_dtype(hamiltonian, prioritize_performance)  # for better performance
        n_partial = len(hamiltonian[0]) // length
        partial_hamiltonian_list = list(zip(*[cp.split(ham, n_partial) for ham in hamiltonian]))

        coefficients_list = None
        if coefficients is not None:
            coefficients = _convert_dtype(coefficients, prioritize_performance)  # type: ignore
            coefficients_list = tuple(cp.split(coefficients, n_partial))

        new_operands: SplittedOperandsDict = {  # type: ignore
            "operands": operands_,
            "partial_hamiltonian_list": partial_hamiltonian_list,
            "hamiltonian_locs": hamiltonian_locs,
            "coefficients_list": coefficients_list,
            "coefficients_loc": coefficients_loc if coefficients_list is not None else None,
        }

    return new_expr, new_operands, new_pname2locs


def _find_dummy_hamiltonian(operands: list[cp.ndarray]) -> list[int]:
    Z = PauliZ(xp=cp)

    locs = []
    for i, op in enumerate(operands):
        if cp.all(op == Z):
            locs.append(i)
    return locs


def _convert_dtype(
    x: cp.ndarray | list[cp.ndarray], prioritize_performance: bool = True
) -> cp.ndarray | list[cp.ndarray]:
    if prioritize_performance:
        if isinstance(x, list):
            return [elem.astype(np.complex64) for elem in x]
        else:
            return x.astype(np.complex64)
    return x


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
