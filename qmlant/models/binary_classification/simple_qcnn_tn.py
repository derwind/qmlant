from __future__ import annotations

from typing import Literal, overload

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qmlant.circuit.library import ZFeatureMap


class SimpleQCNN:
    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        decompose: bool = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        decompose: bool = ...,
        dry_run: Literal[True] = ...,
    ) -> tuple[int, int]:
        ...

    @classmethod
    def make_placeholder_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = False,
        decompose: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | tuple[int, int]:
        """make a Quantum Convolutional Neural Netowrk circuit

        Hierarchical quantum classifiers.
        Iris Cong, Soonwon Choi, Mikhail D. Lukin. Quantum Convolutional Neural Networks. Nature Physics volume 15, pages1273-1278 (2019). https://www.nature.com/articles/s41567-019-0648-8

        simple version: https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html

        Args:
            n_qubits (int): number of qubits
            insert_barrier (bool): insert barriers
            dry_run (bool): True: return only number of needed parameters. False: return a circuit.

        Returns:
            numbers of needed parameters or a circuit
        """

        if dry_run:
            length_feature = cls._make_init_circuit(n_qubits, dry_run=True)
            length_ansatz = cls._make_ansatz(n_qubits, dry_run=True)
            return length_feature, length_ansatz

        qc: QuantumCircuit = cls._make_init_circuit(n_qubits, insert_barrier=insert_barrier)
        ansatz = cls._make_ansatz(n_qubits, insert_barrier=insert_barrier)
        qc.compose(ansatz, inplace=True)
        if decompose:
            qc = qc.decompose()

        return qc

    @classmethod
    def get_hamiltonian(cls, n_qubits: int) -> str:
        """make a Hamiltonian for a Quantum Convolutional Neural Netowrk circuit

        Args:
            n_qubits (int): number of qubits

        Returns:
            a Hamiltonian
        """

        hamiltonian = list("I" * n_qubits)
        hamiltonian[-1] = "Z"
        return "".join(hamiltonian)

    @classmethod
    def _make_init_circuit(
        cls,
        n_qubits: int,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        if dry_run:
            return n_qubits

        init_circuit = ZFeatureMap(n_qubits, insert_barriers=insert_barrier, parameter_multiplier=1)

        return init_circuit

    @overload
    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        dry_run: Literal[False] = ...,
    ) -> QuantumCircuit:
        ...

    @overload
    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = ...,
        dry_run: Literal[True] = ...,
    ) -> int:
        ...

    @classmethod
    def _make_ansatz(
        cls,
        n_qubits: int,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> QuantumCircuit | int:
        if bin(n_qubits)[2:].count("1") != 1:  # should be power of two
            raise ValueError(f"{n_qubits} must be power of two")

        # 2 ** log2(n_qubits) = n_qubits
        conv_params_length = 3 * (2 * (n_qubits - 1))
        pool_params_length = 3 * (n_qubits - 1)
        length = conv_params_length + pool_params_length

        if dry_run:
            return length

        ansatz = QuantumCircuit(n_qubits, name="Ansatz")

        layer_width = n_qubits
        conv_start = 0
        i = 1
        while layer_width > 1:
            qubits = list(range(conv_start, n_qubits))
            ansatz.compose(
                conv_layer(layer_width, f"c{i}", insert_barrier=insert_barrier),
                qubits,
                inplace=True,
            )

            from_qubits, to_qubits = (
                l.tolist() for l in np.array_split(list(range(layer_width)), 2)
            )

            ansatz.compose(
                pool_layer(from_qubits, to_qubits, f"p{i}", insert_barrier=insert_barrier),
                qubits,
                inplace=True,
            )

            layer_width = layer_width // 2
            conv_start += layer_width
            i += 1

        assert ansatz.num_parameters == length, ansatz.num_parameters
        return ansatz


# https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html


def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix, insert_barrier: bool = False):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        if insert_barrier:
            qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        if insert_barrier:
            qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


def pool_layer(sources, sinks, param_prefix, insert_barrier: bool = False):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        if insert_barrier:
            qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc
