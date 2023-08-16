from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, overload

import cupy as cp
import numpy as np
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit

from ..neural_network import Ry_Rydag, Ry_Rydag_direct


@overload
def find_ry_locs(
    qc_pl: QuantumCircuit, hamiltonian: str, return_tn: Literal[False]
) -> dict[str, tuple[int, int]]:
    ...


@overload
def find_ry_locs(
    qc_pl: QuantumCircuit, hamiltonian: str, return_tn: Literal[True]
) -> tuple[dict[str, tuple[int, int]], str, list[cp.ndarray]]:
    ...


def find_ry_locs(
    qc_pl: QuantumCircuit, hamiltonian: str, return_tn: bool = False
) -> dict[str, tuple[int, int]] | tuple[dict[str, tuple[int, int]], str, list[cp.ndarray]]:
    """find Ry (whose name are "x[i]", "θ[i]" etc.) locations in the given placeholder circuit

    Args:
        qc_pl (QuantumCircuit): a given placeholder circuit
        hamiltonian (str): a Hamiltonian
        return_tn (bool): return a TensorNetwork or not

    Returns:
        dict from parameter names to locations, and a TensorNetwork if return_tn is True
    """

    length = len(qc_pl.parameters)
    eps = 0.01
    params = np.arange(eps, np.pi, (np.pi - eps) / length)
    name2param = {pvec.name: p for pvec, p in zip(qc_pl.parameters, params)}
    qc = qc_pl.bind_parameters(params)
    converter = CircuitToEinsum(qc)
    expr, operands = converter.expectation(hamiltonian)

    pname2locs: dict[str, tuple[int, int]] = {}
    for name, p in name2param.items():
        ry, ry_dag = Ry_Rydag(p)
        loc = None
        dag_loc = None
        for i, t in enumerate(operands):
            if cp.allclose(t, ry):
                loc = i
            elif cp.allclose(t, ry_dag):
                dag_loc = i  # i - len(operands)
            if loc and dag_loc:
                pname2locs[name] = (loc, dag_loc)
                break
    if return_tn:
        return pname2locs, expr, operands

    return pname2locs


def replace_by_batch(
    expr: str,
    operands: list[cp.ndarray],
    pname2theta_list: dict[str, list[float] | np.ndarray],
    pname2locs: dict[str, tuple[int, int]],
    batch_symbol: str = "撥",
) -> tuple[str, list[cp.ndarray]]:
    # symbols are: a, b, c, ..., z, A, B, C, ..., 撥
    ins, out = re.split(r"\s*->\s*", expr)
    ins = re.split(r"\s*,\s*", ins)
    for pname, theta_list in pname2theta_list.items():  # e.g. pname[0] = "x[0]"
        batch_and_batch_dag = cp.array([[*Ry_Rydag(theta, xp=np)] for theta in theta_list])
        batch = batch_and_batch_dag[:, 0]
        batch_dag = batch_and_batch_dag[:, 1]
        loc, dag_loc = pname2locs[pname]
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


def replace_ry(
    operands: list[cp.ndarray],
    pname2theta: dict[str, float],
    pname2locs: dict[str, tuple[int, int]],
) -> list[cp.ndarray]:
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        loc, dag_loc = pname2locs[pname]
        Ry_Rydag_direct(theta, operands[loc], operands[dag_loc])

    return operands


def replace_ry_phase_shift(
    operands: list[cp.ndarray],
    pname2theta: dict[str, float],
    pname2locs: dict[str, tuple[int, int]],
    phase_shift_list: Sequence[float] = (np.pi / 2, -np.pi / 2),
) -> list[cp.ndarray]:
    i = 0
    # θ[0]: [π/2, -π/2], θ[1]: [π/2, -π/2], ...
    for pname, theta in pname2theta.items():  # e.g. pname[0] = "θ[0]"
        for phase_shift in phase_shift_list:
            loc, dag_loc = pname2locs[pname]
            Ry_Rydag_direct(theta + phase_shift, operands[loc][i], operands[dag_loc][i])
            i += 1

    return operands
