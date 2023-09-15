import unittest
import warnings
from collections.abc import Callable, Sequence

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
    @staticmethod
    def make_placeholder_circuit(
        ising_dict: dict[tuple[str] | tuple[str, str], float],
        n_reps: int = 1,
        insert_barrier: bool = False,
        dry_run: bool = False,
    ) -> tuple[QuantumCircuit, Callable[[Sequence[float] | np.ndarray], dict[str, float]]] | int:
        n_qubits = 4
        param_names = []

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

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(0)
        qc.h(2)
        qc.cx(2, 3)
        qc.x(2)

        for _ in range(n_reps):
            # H_P
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
                    qc.rzz(theta, ln, rn)

            # H_M
            theta = beta[bi()]
            param_names.append(theta.name)
            qc.rxx(theta, 0, 1)
            theta = beta[bi()]
            param_names.append(theta.name)
            qc.rxx(theta, 0, 1)
            param_names.append(theta.name)
            qc.rxx(theta, 2, 3)
            theta = beta[bi()]
            param_names.append(theta.name)
            qc.ryy(theta, 2, 3)
            theta = beta[bi()]

        def make_pname2theta(params: Sequence[float] | np.ndarray) -> dict[str, float]:
            return dict(zip(param_names, params))

        return qc, make_pname2theta

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
        mixer = SparsePauliOp(["XXII", "YYII", "IIXX", "IIYY"], [1 / 2, 1 / 2, 1 / 2, 1 / 2])
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
