from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class TTN:
    @overload
    @classmethod
    def make_ttn_placeholder_circuit(
        cls,
        n_qubits: int,
        use_mera: bool = ...,
        insert_barrier: bool = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def make_ttn_placeholder_circuit(
        cls,
        n_qubits: int,
        use_mera: bool = ...,
        insert_barrier: bool = ...,
        dry_run: Literal[True] = ...,
    ) -> int:
        ...

    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        use_mera: bool = False,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        """make a Tree TensorNetowrk based quantum circuit

        Hierarchical quantum classifiers.
        Grant, E., Benedetti, M., Cao, S. et al. Hierarchical quantum classifiers. npj Quantum Inf 4, 65 (2018). https://doi.org/10.1038/s41534-018-0116-9

        Args:
            n_qubits (int): number of qubits
            use_mera (bool): use MERA based structure
            insert_barrier (bool): insert barriers
            dry_run (bool): True: return only number of needed parameters. False: return a circuit.

        Returns:
            number of needed parameters or a circuit
        """

        if dry_run:
            length_feature = cls._make_init_circuit(n_qubits, dry_run=True)
            length_ansatz = cls._make_ansatz(n_qubits, use_mera=use_mera, dry_run=True)
            length = length_feature + length_ansatz
            return length

        qc: QuantumCircuit = cls._make_init_circuit(n_qubits)
        ansatz = cls._make_ansatz(n_qubits, use_mera=use_mera, insert_barrier=insert_barrier)
        qc.compose(ansatz, inplace=True)

        return qc

    @classmethod
    def get_hamiltonian(cls, n_qubits: int) -> str:
        """make a Hamiltonian for a Tree TensorNetowrk based quantum circuit

        Hierarchical quantum classifiers.
        Grant, E., Benedetti, M., Cao, S. et al. Hierarchical quantum classifiers. npj Quantum Inf 4, 65 (2018). https://doi.org/10.1038/s41534-018-0116-9

        Args:
            n_qubits (int): number of qubits

        Returns:
            a Hamiltonian
        """

        def calc_power(n, power=0):
            if n % 2 == 1:
                return power
            return calc_power(n // 2, power + 1)

        def zloc(n_qubits):
            n = 0
            for _ in range(calc_power(n_qubits)):
                n = n * 2 + (n + 1) % 2
            return n

        loc = zloc(n_qubits)
        hamiltonian = list("I" * n_qubits)
        hamiltonian[loc] = "Z"
        return "".join(hamiltonian)

    # Eqn. (1)
    @classmethod
    def _make_init_circuit(
        cls,
        n_qubits: int,
        params: Sequence[float] | None = None,
        param_vector: ParameterVector | list | None = None,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        if dry_run:
            return n_qubits

        init_circuit = QuantumCircuit(n_qubits)
        if params is not None:
            for i in range(n_qubits):
                init_circuit.ry(params[i], i)
        elif param_vector is not None:
            for i in range(n_qubits):
                init_circuit.ry(param_vector[i], i)
        else:
            x = ParameterVector("x", n_qubits)
            for i in range(n_qubits):
                init_circuit.ry(x[i], i)

        return init_circuit

    @overload
    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        use_mera: bool = ...,
        param_vector: ParameterVector | list | None = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        use_mera: bool = ...,
        param_vector: ParameterVector | list | None = ...,
        dry_run: Literal[True] = ...,
    ) -> int:
        ...

    # Fig. 1 (b) MERA classifier
    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = False,
        use_mera: bool = False,
        param_vector: ParameterVector | list | None = None,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        def append_U(qc, i, j, thetas, count, last_unitary=False, reverse=False):
            if False:  # pylint: disable=(using-constant-test
                qc.u(thetas[count], thetas[count + 1], thetas[count + 2], i)
                count += 3
                qc.u(thetas[count], thetas[count + 1], thetas[count + 2], j)
                count += 3
            else:
                qc.ry(thetas[count], i)
                count += 1
                qc.ry(thetas[count], j)
                count += 1

            if reverse:
                ansatz.cx(j, i)
            else:
                ansatz.cx(i, j)
            if last_unitary:
                if False:  # pylint: disable=(using-constant-test
                    qc.u(thetas[count], thetas[count + 1], thetas[count + 2], j)
                    count += 3
                else:
                    qc.ry(thetas[count], j)
                    count += 1
            return count

        def append_D(qc, i, j):
            qc.cz(i, j)

        if bin(n_qubits)[2:].count("1") != 1:  # should be power of two
            raise ValueError(f"{n_qubits} must be power of two")

        n_gate_params = 1  # 1 for RYGate, 3 for UGate

        length = (3 * n_qubits // n_qubits) * n_gate_params
        divisor = n_qubits // 2
        while divisor >= 2:
            length += (2 * n_qubits // divisor) * n_gate_params
            divisor = divisor // 2

        if dry_run:
            return length

        if param_vector is None:
            thetas = ParameterVector("θ", length)
        else:
            thetas = param_vector

        count = 0
        ansatz = QuantumCircuit(n_qubits)

        n = 2
        U_start_idx = 0
        U_offset = 1
        p = 1
        while n <= n_qubits // 2:
            # D-block
            if use_mera:
                D_start_index = U_start_idx * 2 + (U_start_idx + 1) % 2  # = next U_start_idx
                D_offset = U_offset * 2 + (-1) ** p  # = next U_offset
                for i in range(D_start_index, n_qubits, n):
                    if i + D_offset >= n_qubits:
                        break
                    append_D(ansatz, i, i + D_offset)
                if insert_barrier:
                    ansatz.barrier()
            # U-block
            reverse = False
            for i in range(U_start_idx, n_qubits, n):
                if i + U_offset >= n_qubits:
                    break
                count = append_U(ansatz, i, i + U_offset, thetas, count, reverse=reverse)
                reverse = not reverse
            if insert_barrier:
                ansatz.barrier()

            U_start_idx = U_start_idx * 2 + (U_start_idx + 1) % 2
            U_offset = U_offset * 2 + (-1) ** p
            p += 1
            n *= 2

        assert n == n_qubits

        # last U
        reverse = False
        for i in range(U_start_idx, n_qubits, n):
            if i + U_offset >= n_qubits:
                break
            count = append_U(ansatz, i, i + U_offset, thetas, count, last_unitary=True)
            reverse = not reverse
        if insert_barrier:
            ansatz.barrier()

        assert count == length, count
        return ansatz
