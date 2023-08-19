import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import math
from collections.abc import Sequence

import cupy as cp
import numpy as np
from torchvision import transforms
from cuquantum import CircuitToEinsum, contract
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qmlant.models.binary_classification import SimpleQCNN
from qmlant.neural_networks import (
    EstimatorTN,
    circuit_to_einsum_expectation,
    replace_by_batch,
    replace_pauli,
)
from qmlant.transforms import MapLabel, ToTensor


class TestQCNN(unittest.TestCase):
    def test_ansatz(self):
        for decompose in [False, True]:
            with self.subTest(decompose=decompose):
                # make effect of barriers to a circuit be same as that of Qiskit codes
                ansatz = SimpleQCNN._make_ansatz(8, insert_barrier=True)
                answer_ansatz = make_qiskit_qcnn_ansatz()
                if decompose:
                    ansatz = ansatz.decompose()
                    answer_ansatz = answer_ansatz.decompose()

                self.assertEqual(ansatz.num_parameters, answer_ansatz.num_parameters)
                param_len = answer_ansatz.num_parameters
                params = [p for p in np.arange(0.01, np.pi, np.pi / param_len)][:param_len]

                converter = CircuitToEinsum(ansatz.bind_parameters(params))
                answer_converter = CircuitToEinsum(answer_ansatz.bind_parameters(params))

                hamiltonian = "I" * 7 + "Z"
                expr, operands = converter.expectation(hamiltonian)
                answer_expr, answer_operands = answer_converter.expectation(hamiltonian)

                self.assertEqual(expr, answer_expr)
                self.assertEqual(len(operands), len(answer_operands))
                for ops, ans_ops in zip(operands, answer_operands):
                    self.assertTrue(cp.all(ops == ans_ops))


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


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
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


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


def make_qiskit_qcnn_ansatz():
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    return ansatz
