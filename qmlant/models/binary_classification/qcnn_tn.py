from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, overload

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qmlant.circuit.library import ZFeatureMap


class QCNN:
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
        decompose: bool = True,
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
    def get_make_pname2theta(
        cls, n_qubits: int
    ) -> Callable[[Sequence[float] | np.ndarray], dict[str, float]]:
        # When layer_width = w,
        # conv_layer requires: (15 * w) params
        # pool_layer requires: (6 * w / 2) params
        def make_pname2theta(params: Sequence[float] | np.ndarray) -> dict[str, float]:
            pname2theta: dict[str, float] = {}

            layer_width = n_qubits
            i = 1
            while layer_width > 1:
                for j in range(15 * layer_width):
                    pname = f"c{i}[{j}]"
                    pname2theta[pname] = params[0]
                    params = params[1:]

                for j in range(6 * layer_width // 2):
                    pname = f"p{i}[{j}]"
                    pname2theta[pname] = params[0]
                    params = params[1:]

                layer_width = layer_width // 2
                i += 1

            assert not params
            return pname2theta

        return make_pname2theta

    @classmethod
    def get_batch_filter(cls) -> Callable[[np.ndarray], np.ndarray]:
        def batch_filter(params: np.ndarray) -> np.ndarray:
            return params * 2  # to match the original ZFeatureMap

        return batch_filter

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

        length = cls.calc_parameters_length(n_qubits)

        if dry_run:
            return length

        ansatz = QuantumCircuit(n_qubits, name="Ansatz")

        layer_width = n_qubits
        conv_start = 0
        i = 1
        while layer_width > 1:
            qubits = list(range(conv_start, n_qubits))
            ansatz.compose(
                cls.conv_layer(layer_width, f"c{i}", insert_barrier=insert_barrier),
                qubits,
                inplace=True,
            )

            from_qubits, to_qubits = (
                l.tolist() for l in np.array_split(list(range(layer_width)), 2)
            )

            ansatz.compose(
                cls.pool_layer(from_qubits, to_qubits, f"p{i}", insert_barrier=insert_barrier),
                qubits,
                inplace=True,
            )

            layer_width = layer_width // 2
            conv_start += layer_width
            i += 1

        assert ansatz.num_parameters == length, ansatz.num_parameters
        return ansatz

    # https://github.com/tensorflow/quantum/blob/master/docs/tutorials/qcnn.ipynb

    @classmethod
    def calc_parameters_length(cls, n_qubits: int):
        # 2 ** log2(n_qubits) = n_qubits
        conv_params_length = 15 * (2 * (n_qubits - 1))
        pool_params_length = 6 * (n_qubits - 1)
        return conv_params_length + pool_params_length

    @classmethod
    def conv_circuit(cls, params):
        target = QuantumCircuit(2)
        target.rx(params[0], 0)
        target.ry(params[1], 0)
        target.rz(params[2], 0)
        target.rx(params[3], 1)
        target.ry(params[4], 1)
        target.rz(params[5], 1)
        target.rzz(params[6], 0, 1)
        target.ryy(params[7], 0, 1)
        target.rxx(params[8], 0, 1)
        target.rx(params[9], 0)
        target.ry(params[10], 0)
        target.rz(params[11], 0)
        target.rx(params[12], 1)
        target.ry(params[13], 1)
        target.rz(params[14], 1)
        return target

    @classmethod
    def conv_layer(cls, num_qubits, param_prefix, insert_barrier: bool = False):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 15)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(cls.conv_circuit(params[param_index : (param_index + 15)]), [q1, q2])
            if insert_barrier:
                qc.barrier()
            param_index += 15
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(cls.conv_circuit(params[param_index : (param_index + 15)]), [q1, q2])
            if insert_barrier:
                qc.barrier()
            param_index += 15

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    @classmethod
    def pool_circuit(cls, params):
        target = QuantumCircuit(2)
        target.rx(params[0], 1)
        target.ry(params[1], 1)
        target.rz(params[2], 1)
        target.rx(params[3], 0)
        target.ry(params[4], 0)
        target.rz(params[5], 0)
        target.cx(0, 1)
        target.rz(-params[2], 1)
        target.ry(-params[1], 1)
        target.rx(-params[0], 1)

        return target

    @classmethod
    def pool_layer(cls, sources, sinks, param_prefix, insert_barrier: bool = False):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 6)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                cls.pool_circuit(params[param_index : (param_index + 6)]), [source, sink]
            )
            if insert_barrier:
                qc.barrier()
            param_index += 6

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc
