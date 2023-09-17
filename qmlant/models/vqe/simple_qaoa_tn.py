from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, overload

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .utils import _calc_num_qubits


class SimpleQAOA:
    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        ising_dict: dict[tuple[str] | tuple[str, str], float],
        n_reps: int = ...,
        insert_barrier: bool = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        ising_dict: dict[tuple[str] | tuple[str, str], float],
        n_reps: int = ...,
        insert_barrier: bool = ...,
        dry_run: Literal[True] = ...,
    ) -> int:
        ...

    @classmethod
    def make_placeholder_circuit(
        cls,
        ising_dict: dict[tuple[str] | tuple[str, str], float],
        n_reps: int = 1,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        """make a SimpleQAOA quantum circuit

        A Quantum Approximate Optimization Algorithm.
        Edward Farhi, Jeffrey Goldstone, Sam Gutmann. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028

        Args:
            ising_dict (dict[tuple[str] | tuple[str, str], float]): a dict defining Ising Hamiltonian
            n_reps (int): number of repetition of blocks
            insert_barrier (bool): insert barriers
            dry_run (bool): True: return only number of needed parameters. False: return a circuit.

        Returns:
            numbers of needed parameters or a circuit
        """

        n_qubits = _calc_num_qubits(ising_dict)
        param_names = []

        if dry_run:
            length_ansatz = (n_qubits + len(ising_dict)) * n_reps
            return length_ansatz

        def rzz(
            qc: QuantumCircuit, theta: float, qubit1: int, qubit2: int, decompose: bool = False
        ):
            if decompose:
                qc.cx(qubit1, qubit2)
                qc.rz(theta, qubit2)
                qc.cx(qubit1, qubit2)
            else:
                qc.rzz(theta, qubit1, qubit2)

        beta = ParameterVector("β", n_qubits * n_reps)
        gamma = ParameterVector("γ", len(ising_dict) * n_reps)
        beta_idx = iter(range(n_qubits * n_reps))

        def bi():
            return next(beta_idx)

        gamma_idx = iter(range(len(ising_dict) * n_reps))

        def gi():
            return next(gamma_idx)

        qc = QuantumCircuit(n_qubits)
        qc.h(qc.qregs[0][:])
        for _ in range(n_reps):
            for k in ising_dict:
                if len(k) == 1:
                    left = k[0]
                    ln = int(left[1:])
                    rn = None
                elif len(k) == 2:
                    left, right = k  # type: ignore
                    ln = int(left[1:])
                    rn = int(right[1:])
                    assert ln <= rn
                else:
                    raise ValueError(f"len(k) = {len(k)} must be one or two.")

                if rn is None:
                    theta = gamma[gi()]
                    param_names.append(theta.name)
                    qc.rz(theta, ln)
                else:
                    theta = gamma[gi()]
                    param_names.append(theta.name)
                    rzz(qc, theta, ln, rn)
            if insert_barrier:
                qc.barrier()
            for i in range(n_qubits):
                theta = beta[bi()]
                param_names.append(theta.name)
                qc.rx(theta, i)
            if insert_barrier:
                qc.barrier()

        return qc
