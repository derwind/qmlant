import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram

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

    def test_circuit_to_einsum_expectation_with_coefficients(self):
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
        coefficients = [2, -3, 1, -1.5, 5.1, 0.3, -4.3, -0.5]
        hamiltonian = SparsePauliOp(pauli_list, coefficients)
        estimator = Estimator()
        result = estimator.run([qc], [hamiltonian], init).result()
        answer_expval = result.values[0]

        hamiltonian_tn = self.pauli_list2hamiltonian(pauli_list)
        expr, operands, pname2locs = circuit_to_einsum_expectation(qc, hamiltonian_tn, coefficients)
        estimator_tn = EstimatorTN(pname2locs, expr, operands)
        expval = estimator_tn.forward(init)
        self.assertAlmostEqual(expval, answer_expval, 5)

    def test_circuit_to_einsum_expectation_with_coefficients2(self):
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
        coefficients = [2, -3, 1, -1.5, 5.1, 0.3, -4.3, -0.5]
        hamiltonian = SparsePauliOp(pauli_list, coefficients)
        hamiltonian_tn = self.pauli_list2hamiltonian(pauli_list)
        rng = np.random.default_rng(42)
        inits = np.split(
            rng.random(qc.num_parameters * (len(pauli_list) - 1)) * 2 * np.pi,
            len(pauli_list) - 1,
        )

        for i, init in enumerate(inits):
            estimator = Estimator()
            result = estimator.run([qc], [hamiltonian], init).result()
            answer_expval = result.values[0]

            n_partial_hamiltonian = i + 2
            expr, operands, pname2locs = circuit_to_einsum_expectation(
                qc, hamiltonian_tn, coefficients, n_partial_hamiltonian=n_partial_hamiltonian
            )
            estimator_tn = EstimatorTN(pname2locs, expr, operands)
            expval = estimator_tn.forward(init)
            self.assertAlmostEqual(expval, answer_expval, 5)


class TestQAOA(unittest.TestCase):
    def test_qaoa_xy_mixer(self):
        linear = {"q0": 4.0, "q1": 4.0, "q2": 4.0, "q3": 4.0}
        quadratic = {
            ("q1", "q3"): 2.0,
            ("q2", "q3"): 2.0,
            ("q0", "q1"): 4.0,
            ("q0", "q2"): 4.0,
            ("q1", "q2"): 8.0,
        }

        qubo = QuadraticProgram()
        qubo.binary_var("q0")
        qubo.binary_var("q1")
        qubo.binary_var("q2")
        qubo.binary_var("q3")

        qubo.minimize(linear=linear, quadratic=quadratic)
        qubit_op, offset = qubo.to_ising()

        initial_state_circuit = QuantumCircuit(4)
        initial_state_circuit.h(0)
        initial_state_circuit.cx(0, 1)
        initial_state_circuit.x(0)
        initial_state_circuit.h(2)
        initial_state_circuit.cx(2, 3)
        initial_state_circuit.x(2)

        sampler = Sampler()
        optimizer = COBYLA()
        step = 1
        mixer = SparsePauliOp(["XXII", "YYII", "IIXX", "IIYY"], [1/2, 1/2, 1/2, 1/2])
        qaoa = QAOA(
            sampler,
            optimizer,
            reps=step,
            initial_state=initial_state_circuit,
            mixer=mixer,
        )
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        answer = result.best_measurement["bitstring"]
        self.assertEqual(answer, "1001")
