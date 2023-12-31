from __future__ import annotations

from typing import Literal, overload

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
            length_ansatz = 2 * n_reps
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

        betas = ParameterVector("β", n_reps)
        beta_idx = iter(range(n_reps))

        def bi():
            return next(beta_idx)

        gammas = ParameterVector("γ", n_reps)
        gamma_idx = iter(range(n_reps))

        def gi():
            return next(gamma_idx)

        qc = QuantumCircuit(n_qubits)
        qc.h(qc.qregs[0][:])
        for _ in range(n_reps):
            # H_P
            gamma = gammas[gi()]
            param_names.append(gamma.name)

            for k in ising_dict:
                if len(k) == 1:
                    left = k[0]
                    ln = int(left[1:])
                    qc.rz(gamma, ln)
                elif len(k) == 2:
                    left, right = k  # type: ignore
                    ln = int(left[1:])
                    rn = int(right[1:])
                    assert ln <= rn
                    rzz(qc, gamma, ln, rn)
                else:
                    raise ValueError(f"len(k) = {len(k)} must be one or two.")

            if insert_barrier:
                qc.barrier()

            # H_M
            beta = betas[bi()]
            param_names.append(beta.name)

            for i in range(n_qubits):
                qc.rx(beta, i)
            if insert_barrier:
                qc.barrier()

        return qc
