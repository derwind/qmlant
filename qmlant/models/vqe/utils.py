from __future__ import annotations

from collections.abc import Sequence

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit

from qmlant.neural_networks.utils import (
    Identity,
    MatZero,
    OperandsDict,
    ParameterName2Locs,
    Pauli,
    PauliLocs,
    PauliZ,
    SplittedOperandsDict,
)
from qmlant.neural_networks.utils import (
    circuit_to_einsum_expectation as nnu_circuit_to_einsum_expectation,
)
from qmlant.neural_networks.utils.pauli import Rz_Rzdag, Rzz_Rzzdag


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
            tuple[list[cp.ndarray]]: Hamiltonian array, e.g., Z_1 Z_2 + Z_3 Z_4
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

    @staticmethod
    def to_pauli_strings(hamiltonian: list[cp.ndarray]):
        I = Identity(xp=cp)  # noqa: E741
        Z = PauliZ(xp=cp)

        pauli_strings: list[str] = []
        for ham in zip(*hamiltonian):
            pauli_str = ""
            for h in ham:
                if cp.allclose(h, I):
                    pauli_str += "I"
                elif cp.allclose(h, Z):
                    pauli_str += "Z"
                else:
                    raise ValueError(f"{h} must be I or Z.")
            pauli_strings.append(pauli_str)
        return pauli_strings


def circuit_to_einsum_expectation(
    qc_pl: QuantumCircuit,
    hamiltonian: list[cp.ndarray],
    coefficients: np.ndarray | None = None,
    is_qaoa: bool = False,
    prioritize_performance: bool = True,
    n_partial_hamiltonian: int = 1,
) -> (
    tuple[str, OperandsDict, ParameterName2Locs]
    | tuple[str, SplittedOperandsDict, ParameterName2Locs]
):
    """CircuitToEinsum with hamiltonian embedded for `expectation`

    Args:
        qc_pl (QuantumCircuit): placeholder `QuantumCircuit` with `ParameterVector`
        hamiltonian (list[cp.ndarray]): Hamiltonian list
        coefficients (np.ndarray | None): coefficients of Hamiltonian
        is_qaoa (bool): set `True` if QAOA circuit
        prioritize_performance (bool): use `np.complex64` instead of `np.complex128`
        n_partial_hamiltonian (int): III+IIZ, IZI,ZII, ZIZ+ZZI for
            III + IIZ + IZI + ZII + ZIZ + ZZI if n_partial_hamiltonian = 2

    Returns:
        str: `expr`
        list[cp.ndarray] | tuple[list[int], tuple[tuple[cp.ndarray]], list[int]]: `operands`,
            or tuple of `operands`, list of partial hamitonian list and hamiltonian locs
        dict[str, tuple[list[int], list[int], Pauli]]: dict of parameter name to locs
    """

    def make_pname2theta(params: Sequence[float] | np.ndarray) -> dict[str, float]:
        return dict(zip([param.name for param in qc_pl.parameters], params))

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
        # update TN
        es.insert(coefficients_loc, "食")
        # shift +1 for dag_locs due to embedding of coefficients into TN
        for name, pauli2locs in pname2locs.items():
            new_pauli2locs: dict[Pauli, PauliLocs] = {}
            for make_paulis, (locs, dag_locs, coeffs) in pauli2locs.items():
                shifted_dag_locs = [v + 1 for v in dag_locs]
                new_pauli2locs[make_paulis] = PauliLocs(locs, shifted_dag_locs, coeffs)
            new_pname2locs[name] = new_pauli2locs
        coefficients = cp.array(coefficients, dtype=complex)
        operands.insert(coefficients_loc, coefficients)

        if is_qaoa:
            # update pname2locs (especially `PauliLocs.coefficients`)
            # according to coefficients of the problem Hamiltonian
            pauli_str2coeff = _make_pauli_str2coeff(hamiltonian, coefficients)
            char2qubits = _make_char2qubits(expr, operands, min(hamiltonian_locs))
            new_pname2locs = _update_pname2locs(
                expr, qc_pl.num_qubits, char2qubits, pauli_str2coeff, new_pname2locs
            )

    new_expr = ",".join(es) + "->"

    # update operands with real hamiltonian
    if n_partial_hamiltonian <= 1:
        for ham, locs in zip(hamiltonian, hamiltonian_locs):  # type: ignore
            operands[locs] = ham  # type: ignore
        operands_ = _convert_dtype(operands, prioritize_performance)  # for better performance
        new_operands: SplittedOperandsDict = {  # type: ignore
            "operands": operands_,
            "make_pname2theta": make_pname2theta,
        }
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
            "make_pname2theta": make_pname2theta,
            "partial_hamiltonian_list": partial_hamiltonian_list,
            "hamiltonian_locs": hamiltonian_locs,
            "coefficients_list": coefficients_list,
            "coefficients_loc": coefficients_loc if coefficients_list is not None else None,
        }

    return new_expr, new_operands, new_pname2locs


def _update_pname2locs(
    expr: str,
    n_qubits: int,
    char2qubits: dict[str, int],
    pauli_str2coeff: dict[str, float],
    pname2locs: ParameterName2Locs,
) -> ParameterName2Locs:
    indices = expr.split("->")[0].split(",")

    for _, pauli_locs in pname2locs.items():
        for op, (locs, _, pauli_coefficients) in pauli_locs.items():
            if op == Rz_Rzdag:
                for i, loc in enumerate(locs):
                    c0 = list(indices[loc])[0]  # "d", "ea", "fgbe" etc.
                    qubit0 = char2qubits[c0]
                    pauli_str = ["I"] * n_qubits
                    pauli_str[qubit0] = "Z"
                    pauli_str_ = "".join(pauli_str)
                    updated_coeff = pauli_str2coeff[pauli_str_] * 2
                    pauli_coefficients[i] = updated_coeff
            elif op == Rzz_Rzzdag:
                for i, loc in enumerate(locs):
                    c1, c0 = list(indices[loc])[:2]  # "d", "ea", "fgbe" etc.
                    qubit0 = char2qubits[c0]
                    qubit1 = char2qubits[c1]
                    pauli_str = ["I"] * n_qubits
                    pauli_str[qubit0] = pauli_str[qubit1] = "Z"
                    pauli_str_ = "".join(pauli_str)
                    updated_coeff = pauli_str2coeff[pauli_str_] * 2
                    pauli_coefficients[i] = updated_coeff
            else:
                for i, _ in enumerate(locs):
                    updated_coeff = 2.0
                    pauli_coefficients[i] = updated_coeff

    return pname2locs


def _make_pauli_str2coeff(
    hamiltonian: list[cp.ndarray], coefficients: cp.ndarray
) -> dict[str, float]:
    pauli_strings = HamiltonianConverter.to_pauli_strings(hamiltonian)
    pauli_str2coeff = dict(zip(pauli_strings, coefficients))
    return pauli_str2coeff


def _make_char2qubits(
    expr: str, operands: list[cp.ndarray], min_hamiltonian_locs: int
) -> dict[str, int]:
    """determine which qubit is associated with the character contained in `expr`"""

    ZERO = cp.array([1, 0], dtype=complex)
    indices = expr.split("->")[0].split(",")

    char2qubits: dict[str, int] = {}
    qubits = 0

    for i, (ex, op) in enumerate(zip(indices, operands)):
        if i >= min_hamiltonian_locs:
            break
        if op.shape == (2,):
            if cp.allclose(op, ZERO):
                char2qubits[ex] = qubits
                qubits += 1
        elif op.shape == (2, 2):
            to_, from_ = list(ex)
            char2qubits[to_] = char2qubits[from_]
        elif op.shape == (2, 2, 2, 2):
            to_1, to_0, from_1, from_0 = list(ex)
            char2qubits[to_0] = char2qubits[from_0]
            char2qubits[to_1] = char2qubits[from_1]

    return char2qubits


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
