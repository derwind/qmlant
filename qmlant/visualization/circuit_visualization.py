from __future__ import annotations

import re

import networkx as nx
import numpy as np
import quimb.tensor as qtn
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .circuit_converter import circuit_to_quimb_tn


def get_graph(
    circuit: str | QuantumCircuit,
    hamiltonian: str | None = None,
    assign_dummy_parameters: bool = False,
) -> nx.Graph:
    """Create a network graph

    Args:
        circuit (str | QuantumCircuit): `expr` of cuTensorNet or `QuantumCircuit` of Qiskit

    Return:
        a graph
    """

    if isinstance(circuit, QuantumCircuit):
        if assign_dummy_parameters:
            qc = circuit.bind_parameters([0] * len(circuit.parameters))

        converter = CircuitToEinsum(qc)
        if hamiltonian is None:
            expr, _ = converter.state_vector()
        else:
            expr, _ = converter.expectation(hamiltonian)
    else:
        expr = circuit

    expr = re.sub(r"\s*->\s*", ",", expr)
    if expr.endswith(","):
        expr = expr[:-1]

    edgelist = []
    nodes = expr.split(",")
    for i in range(len(nodes) - 1):
        for c in nodes[i]:
            for j in range(i + 1, len(nodes)):
                if c in nodes[j]:
                    edgelist.append((i, j))
    return nx.from_edgelist(edgelist)


def draw_graph(
    circuit: str | QuantumCircuit,
    hamiltonian: str | None = None,
    assign_dummy_parameters: bool = False,
) -> None:
    G: nx.Graph = get_graph(circuit, hamiltonian, assign_dummy_parameters)
    nx.draw(G)


def get_quimb_tn(
    circuit: QuantumCircuit,
    hamiltonian: str | SparsePauliOp | None = None,
    assign_dummy_parameters: bool = False,
) -> qtn.Circuit:
    if assign_dummy_parameters:
        length = len(circuit.parameters)
        eps = 0.01
        params = np.arange(eps, np.pi - eps, (np.pi - 2 * eps) / length)[:length]
        state = circuit.bind_parameters(params)
    else:
        state = circuit
    qc = state.copy()

    if hamiltonian is not None:
        if isinstance(hamiltonian, SparsePauliOp):
            hamiltonian = [op[::-1] for op in hamiltonian.paulis.to_labels()]
        else:
            hamiltonian = [hamiltonian]

        for operator in hamiltonian:
            for i, op_ in enumerate(operator):
                op = op_.upper()
                if op == "I":
                    continue

                if op == "X":
                    qc.x(i)
                elif op == "Y":
                    qc.y(i)
                elif op == "Z":
                    qc.z(i)
                else:
                    print(f"Not implemented for an operator: {op}")

        qc.compose(state.inverse(), inplace=True)

    return circuit_to_quimb_tn(qc)


def draw_quimb_tn(
    circuit: QuantumCircuit,
    hamiltonian: str | SparsePauliOp | None = None,
    assign_dummy_parameters: bool = False,
) -> None:
    tn = get_quimb_tn(circuit, hamiltonian, assign_dummy_parameters)

    color = [f"I{n}" for n in range(circuit.num_qubits)]
    tn.psi.draw(color=color)
