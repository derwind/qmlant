from __future__ import annotations

import re
from collections.abc import Sequence

import cupy as cp
import numpy as np
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit

from .pauli import Pauli, Rx_Rxdag, Rx_Rxdag_Ry_Rydag_Rz_Rzdag, Ry_Rydag, Rz_Rzdag


def circuit_to_einsum_expectation(
    qc_pl: QuantumCircuit, hamiltonian: str
) -> tuple[str, list[cp.ndarray], dict[str, tuple[int, int, Pauli]]]:
    """apply CircuitToEinsum and find Pauli (whose name are "x[i]", "θ[i]" etc.) locations in the given placeholder circuit

    Args:
        qc_pl (QuantumCircuit): a given placeholder circuit
        hamiltonian (str): a Hamiltonian

    Returns:
        a TensorNetwork and dict from parameter names to locations
    """

    length = len(qc_pl.parameters)
    eps = 0.01
    params = np.arange(eps, np.pi, (np.pi - eps) / length)
    name2param = {pvec.name: p for pvec, p in zip(qc_pl.parameters, params)}
    qc = qc_pl.bind_parameters(params)
    converter = CircuitToEinsum(qc)
    expr, operands = converter.expectation(hamiltonian)

    pname2locs: dict[str, tuple[int, int, Pauli]] = {}
    for name, p in name2param.items():
        rx, rx_dag, ry, ry_dag, rz, rz_dag = Rx_Rxdag_Ry_Rydag_Rz_Rzdag(p)
        loc = None
        dag_loc = None
        make_paulis: Pauli = None
        for i, t in enumerate(operands):
            if cp.allclose(t, rx):
                loc = i
                make_paulis = Rx_Rxdag
            elif cp.allclose(t, rx_dag):
                dag_loc = i  # i - len(operands)
            elif cp.allclose(t, ry):
                loc = i
                make_paulis = Ry_Rydag
            elif cp.allclose(t, ry_dag):
                dag_loc = i  # i - len(operands)
            elif cp.allclose(t, rz):
                loc = i
                make_paulis = Rz_Rzdag
            elif cp.allclose(t, rz_dag):
                dag_loc = i  # i - len(operands)
            if loc and dag_loc:
                pname2locs[name] = (loc, dag_loc, make_paulis)
                break
    return expr, operands, pname2locs


def replace_by_batch(
    expr: str,
    operands: list[cp.ndarray],
    pname2theta_list: dict[str, list[float] | np.ndarray],
    pname2locs: dict[str, tuple[int, int, Pauli]],
    batch_symbol: str = "撥",
) -> tuple[str, list[cp.ndarray]]:
    # symbols are: a, b, c, ..., z, A, B, C, ..., 撥
    ins, out = re.split(r"\s*->\s*", expr)
    ins = re.split(r"\s*,\s*", ins)
    for pname, theta_list in pname2theta_list.items():  # e.g. pname[0] = "x[0]"
        loc, dag_loc, make_paulis = pname2locs[pname]
        batch_and_batch_dag = cp.array([[*make_paulis(theta, xp=np)] for theta in theta_list])
        batch = batch_and_batch_dag[:, 0]
        batch_dag = batch_and_batch_dag[:, 1]
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
    pname2locs: dict[str, tuple[int, int, Pauli]],
) -> list[cp.ndarray]:
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        loc, dag_loc, make_paulis = pname2locs[pname]
        make_paulis(theta, operands[loc], operands[dag_loc])

    return operands


def replace_pauli_phase_shift(
    operands: list[cp.ndarray],
    pname2theta: dict[str, float],
    pname2locs: dict[str, tuple[int, int, Pauli]],
    phase_shift_list: Sequence[float] = (np.pi / 2, -np.pi / 2),
) -> list[cp.ndarray]:
    i = 0
    # θ[0]: [π/2, -π/2], θ[1]: [π/2, -π/2], ...
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        for phase_shift in phase_shift_list:
            loc, dag_loc, make_paulis = pname2locs[pname]
            make_paulis(theta + phase_shift, operands[loc][i], operands[dag_loc][i])
            i += 1

    return operands
