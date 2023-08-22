from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .qcnn_tn import QCNN


class SimpleQCNN(QCNN):
    @classmethod
    def get_make_pname2theta(
        cls, n_qubits: int
    ) -> Callable[[Sequence[float] | np.ndarray], dict[str, float]]:
        # When layer_width = w,
        # conv_layer requires: (3 * w) params
        # pool_layer requires: (3 * w / 2) params
        def make_pname2theta(params: Sequence[float] | np.ndarray) -> dict[str, float]:
            pname2theta: dict[str, float] = {}

            layer_width = n_qubits
            i = 1
            while layer_width > 1:
                for j in range(3 * layer_width):
                    pname = f"c{i}[{j}]"
                    pname2theta[pname] = params[0]
                    params = params[1:]

                for j in range(3 * layer_width // 2):
                    pname = f"p{i}[{j}]"
                    pname2theta[pname] = params[0]
                    params = params[1:]

                layer_width = layer_width // 2
                i += 1

            assert not params
            return pname2theta

        return make_pname2theta

    @classmethod
    def calc_parameters_length(cls, n_qubits: int):
        # 2 ** log2(n_qubits) = n_qubits
        conv_params_length = 3 * (2 * (n_qubits - 1))
        pool_params_length = 3 * (n_qubits - 1)
        return conv_params_length + pool_params_length

    @classmethod
    def conv_circuit(cls, params):
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

    @classmethod
    def conv_layer(cls, num_qubits, param_prefix, insert_barrier: bool = False):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(cls.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            if insert_barrier:
                qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(cls.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            if insert_barrier:
                qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    @classmethod
    def pool_circuit(cls, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target

    @classmethod
    def pool_layer(cls, sources, sinks, param_prefix, insert_barrier: bool = False):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                cls.pool_circuit(params[param_index : (param_index + 3)]), [source, sink]
            )
            if insert_barrier:
                qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc
