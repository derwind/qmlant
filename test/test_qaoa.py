import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from qmlant.models.vqe import circuit_to_einsum_expectation
from qmlant.neural_networks.estimator_tn import EstimatorTN
from qmlant.neural_networks.utils import Identity, PauliZ


class TestExpectation(unittest.TestCase):
    @staticmethod
    def pauli_list2hamiltonian(pauli_list_qiskit: list[str], reverse: bool = True):
        I = Identity()  # noqa: E741
        Z = PauliZ()

        if reverse:
            pauli_list_qiskit = [s[::-1] for s in pauli_list_qiskit]

        hamiltonian_tn = []
        for pauli_chars in zip(*[[c for c in paulis] for paulis in pauli_list_qiskit]):
            hamiltonian_tn.append(cp.array([Z if c == "Z" else I for c in pauli_chars]))
        return hamiltonian_tn

    def test_circuit_to_einsum_expectation(self):
        qc = QuantumCircuit(3)
        theta = ParameterVector("θ", 6)
        qc.rx(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.rz(theta[2], 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        qc.rx(theta[3], 0)
        qc.ry(theta[4], 1)
        qc.rz(theta[5], 2)

        rng = np.random.default_rng(42)
        init = rng.random(qc.num_parameters) * 2 * np.pi

        pauli_list = ["III", "IIZ", "IZI", "IZZ", "ZII", "ZIZ", "ZZI", "ZZZ"]
        hamiltonian = SparsePauliOp(pauli_list)
        estimator = Estimator()
        result = estimator.run([qc], [hamiltonian], init).result()
        answer_expval = result.values[0]

        hamiltonian_tn = self.pauli_list2hamiltonian(pauli_list)
        expr, operands, pname2locs = circuit_to_einsum_expectation(qc, hamiltonian_tn)
        estimator_tn = EstimatorTN(pname2locs, expr, operands)
        expval = estimator_tn.forward(init)
        self.assertAlmostEqual(expval, answer_expval, 6)

    def test_circuit_to_einsum_expectation2(self):
        qc = QuantumCircuit(3)
        theta = ParameterVector("θ", 6)
        qc.rx(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.rz(theta[2], 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        qc.rx(theta[3], 0)
        qc.ry(theta[4], 1)
        qc.rz(theta[5], 2)

        pauli_list = ["III", "IIZ", "IZI", "IZZ", "ZII", "ZIZ", "ZZI", "ZZZ"]
        rng = np.random.default_rng(42)
        inits = np.split(
            rng.random(qc.num_parameters * (len(pauli_list) - 1)) * 2 * np.pi,
            len(pauli_list) - 1,
        )

        for i, init in enumerate(inits):
            hamiltonian = SparsePauliOp(pauli_list)
            estimator = Estimator()
            result = estimator.run([qc], [hamiltonian], init).result()
            answer_expval = result.values[0]

            hamiltonian_tn = self.pauli_list2hamiltonian(pauli_list)
            n_partial_hamiltonian = i + 2
            expr, operands, pname2locs = circuit_to_einsum_expectation(
                qc, hamiltonian_tn, n_partial_hamiltonian=n_partial_hamiltonian
            )
            estimator_tn = EstimatorTN(pname2locs, expr, operands)
            expval = estimator_tn.forward(init)
            self.assertAlmostEqual(expval, answer_expval, 6)
