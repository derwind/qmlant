from __future__ import annotations

import re

import networkx as nx
from cuquantum import CircuitToEinsum
from qiskit import QuantumCircuit


def get_graph(circuit: str | QuantumCircuit, hamiltonian: str | None = None) -> nx.Graph:
    """Create a network graph

    Args:
        circuit (str | QuantumCircuit): `expr` of cuTensorNet or `QuantumCircuit` of Qiskit

    Return:
        a graph
    """

    if isinstance(circuit, QuantumCircuit):
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
