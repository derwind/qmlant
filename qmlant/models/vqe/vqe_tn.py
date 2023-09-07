from __future__ import annotations

from typing import Literal, overload

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class VQE:
    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        dry_run: Literal[True] = ...,
    ) -> int:
        ...

    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        """make a simple VQE quantum circuit

        Args:
            n_qubits (int): number of qubits
            insert_barrier (bool): insert barriers
            dry_run (bool): True: return only number of needed parameters. False: return a circuit.

        Returns:
            numbers of needed parameters or a circuit
        """

        if dry_run:
            length_ansatz = n_qubits
            return length_ansatz

        qc: QuantumCircuit = QuantumCircuit(n_qubits)

        theta = ParameterVector("θ", n_qubits)

        for i in range(n_qubits):
            qc.rx(theta[i], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if insert_barrier:
            qc.barrier()

        return qc
