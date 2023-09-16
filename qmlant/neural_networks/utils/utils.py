from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple, TypedDict

import cupy as cp
import numpy as np
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit

from .pauli import (
    Pauli,
    PauliMatrices,
    Rx_Rxdag,
    Rxx_Rxxdag,
    Ry_Rydag,
    Ryy_Ryydag,
    Rz_Rzdag,
    Rzz_Rzzdag,
)


class SplittedOperandsDict(TypedDict):
    operands: list[cp.ndarray]
    partial_hamiltonian_list: tuple[tuple[cp.ndarray]]
    hamiltonian_locs: list[int]
    coefficients_list: tuple[cp.ndarray] | None
    coefficients_loc: int | None


class PauliLocs(NamedTuple):
    locs: list[int]
    dag_locs: list[int]
    coefficients: list[float]


ParameterName2Locs = dict[str, dict[Pauli, PauliLocs]]


def circuit_to_einsum_expectation(
    qc_pl: QuantumCircuit, hamiltonian: str
) -> tuple[str, list[cp.ndarray], ParameterName2Locs]:
    """apply CircuitToEinsum and find Pauli (whose name are "x[i]", "θ[i]" etc.) locations in the given placeholder circuit

    Args:
        qc_pl (QuantumCircuit): a given placeholder circuit with `ParameterVectorElement`
        hamiltonian (str): a Hamiltonian

    Returns:
        a TensorNetwork and dict from parameter names to locations
    """

    length = len(qc_pl.parameters)
    eps = 0.01
    params = np.arange(eps, np.pi - eps, (np.pi - 2 * eps) / length)[:length]
    name2param = {pvec.name: p for pvec, p in zip(qc_pl.parameters, params)}
    qc = qc_pl.bind_parameters(params)
    converter = CircuitToEinsum(qc)
    expr, operands = converter.expectation(hamiltonian)

    pname2locs: ParameterName2Locs = {}
    for name, param in name2param.items():
        rx, _, ry, _, rz, _, rxx, _, ryy, _, rzz, _ = PauliMatrices(param)
        # consider the possibitity of same parameters are encoded in multiple locations
        # and/or have multiple matrices of different types
        pauli2locs: dict[Pauli, PauliLocs] = {}
        len_operands = len(operands)
        for i, t in enumerate(operands):
            if i >= len_operands / 2:
                break

            if t.shape == (2, 2):
                if cp.allclose(t, ry):
                    make_paulis = Ry_Rydag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
                elif cp.allclose(t, rz):
                    make_paulis = Rz_Rzdag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
                elif cp.allclose(t, rx):
                    make_paulis = Rx_Rxdag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
            elif t.shape == (2, 2, 2, 2):
                if cp.allclose(t, rzz):
                    make_paulis = Rzz_Rzzdag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
                elif cp.allclose(t, rxx):
                    make_paulis = Rxx_Rxxdag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
                elif cp.allclose(t, ryy):
                    make_paulis = Ryy_Ryydag
                    pauli2locs.setdefault(make_paulis, PauliLocs([], [], []))
                    pauli2locs[make_paulis].locs.append(i)
                    pauli2locs[make_paulis].dag_locs.append(len_operands - i - 1)
                    pauli2locs[make_paulis].coefficients.append(1.0)
        if pauli2locs:
            pname2locs[name] = pauli2locs
    return expr, operands, pname2locs


def replace_by_batch(
    expr: str,
    operands: list[cp.ndarray],
    pname2theta_list: dict[str, list[float] | np.ndarray],
    pname2locs: ParameterName2Locs,
    batch_symbol: str = "撥",
) -> tuple[str, list[cp.ndarray]]:
    # symbols are: a, b, c, ..., z, A, B, C, ..., 撥
    ins, out = re.split(r"\s*->\s*", expr)
    ins = re.split(r"\s*,\s*", ins)
    for pname, theta_list in pname2theta_list.items():  # e.g. pname[0] = "x[0]"
        for make_paulis, (locs, dag_locs, _) in pname2locs[pname].items():
            batch_and_batch_dag = cp.array([[*make_paulis(theta, xp=np)] for theta in theta_list])
            batch = batch_and_batch_dag[:, 0]
            batch_dag = batch_and_batch_dag[:, 1]

            for loc, dag_loc in zip(locs, dag_locs):
                operands[loc] = batch
                operands[dag_loc] = batch_dag
                if len(ins[loc]) == 2:
                    ins[loc] = batch_symbol + ins[loc]
                if len(ins[dag_loc]) == 2:
                    ins[dag_loc] = batch_symbol + ins[dag_loc]
    if len(out) == 0:
        out = batch_symbol
    new_expr = ",".join(ins) + "->" + out

    return new_expr, operands


def replace_pauli(
    operands: list[cp.ndarray],
    pname2theta: dict[str, float],
    pname2locs: ParameterName2Locs,
) -> list[cp.ndarray]:
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        # pname may be not found due to cancellation between op and op_dagger
        if pname not in pname2locs:
            continue
        for make_paulis, (locs, dag_locs, coefficients) in pname2locs[pname].items():
            for loc, dag_loc, coeff in zip(locs, dag_locs, coefficients):
                make_paulis(theta * coeff, operands[loc], operands[dag_loc])

    return operands


def replace_pauli_phase_shift(
    operands: list[cp.ndarray],
    pname2theta: dict[str, float],
    pname2locs: ParameterName2Locs,
    phase_shift_list: Sequence[float] = (np.pi / 2, -np.pi / 2),
) -> list[cp.ndarray]:
    i = 0
    # θ[0]: [π/2, -π/2], θ[1]: [π/2, -π/2], ...
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        for phase_shift in phase_shift_list:
            for make_paulis, (locs, dag_locs, coefficients) in pname2locs[pname].items():
                for loc, dag_loc, coeff in zip(locs, dag_locs, coefficients):
                    make_paulis(theta * coeff + phase_shift, operands[loc][i], operands[dag_loc][i])
            i += 1

    return operands
