import unittest
import warnings
from collections.abc import Callable, Sequence

warnings.simplefilter("ignore", DeprecationWarning)

import cupy as cp
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA

from qmlant.models.vqe import HamiltonianConverter, QAOAMixer, circuit_to_einsum_expectation
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
    def make_initial_state_circuit():
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(0)
        qc.h(2)
        qc.cx(2, 3)
        qc.x(2)
        return qc

    @staticmethod
    def make_placeholder_circuit(
        ising_dict: dict[tuple[str] | tuple[str, str], float],
        n_reps: int = 1,
        insert_barrier: bool = ...,
    ) -> tuple[QuantumCircuit, Callable[[Sequence[float] | np.ndarray], dict[str, float]]] | int:
        param_names = []

        betas = ParameterVector("β", n_reps)
        beta_idx = iter(range(n_reps))

        def bi():
            return next(beta_idx)

        gammas = ParameterVector("γ", n_reps)
        gamma_idx = iter(range(n_reps))

        def gi():
            return next(gamma_idx)

        qc = TestQAOA.make_initial_state_circuit()

        if insert_barrier:
            qc.barrier()

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
                    qc.rzz(gamma, ln, rn)
                else:
                    raise ValueError(f"len(k) = {len(k)} must be one or two.")

            if insert_barrier:
                qc.barrier()

            # H_M
            beta = betas[bi()]
            param_names.append(beta.name)

            qc.rxx(beta, 0, 1)
            qc.ryy(beta, 0, 1)
            qc.rxx(beta, 2, 3)
            qc.ryy(beta, 2, 3)

            if insert_barrier:
                qc.barrier()

        return qc

    def test_qaoa_xy_mixer(self):
        ising_dict = {
            ("z0",): -4.0,
            ("z1",): -5.5,
            ("z0", "z1"): 1.0,
            ("z2",): -5.5,
            ("z0", "z2"): 1.0,
            ("z3",): -3.0,
            ("z1", "z2"): 2.0,
            ("z1", "z3"): 0.5,
            ("z2", "z3"): 0.5,
        }
        hamiltonian, coefficients = HamiltonianConverter(ising_dict).get_hamiltonian()
        qubit_op = SparsePauliOp(
            [ham[::-1] for ham in HamiltonianConverter.to_pauli_strings(hamiltonian)],
            coefficients,
        )
        mixer = SparsePauliOp(["XXII", "YYII", "IIXX", "IIYY"], [1/2, 1/2, 1/2, 1/2])

        n_reps = 2

        sampler = Sampler()
        optimizer = COBYLA()
        qaoa = QAOA(
            sampler,
            optimizer,
            reps=n_reps,
            initial_state=TestQAOA.make_initial_state_circuit(),
            mixer=mixer,
        )
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        answer = result.best_measurement["bitstring"]
        self.assertEqual(answer, "1001")

        qc = TestQAOA.make_placeholder_circuit(ising_dict, n_reps=n_reps)
        expr, operands, pname2locs = circuit_to_einsum_expectation(
            qc,
            hamiltonian,
            coefficients,
            qaoa_mixer=QAOAMixer.XY_MIXER,
        )

        estimator = EstimatorTN(pname2locs, expr, operands)

        def comnpute_expectation_tn(params, *args):
            (estimator,) = args
            return estimator.forward(params)

        rng = np.random.default_rng(42)
        init = rng.random(qc.num_parameters) * 2*np.pi

        result = minimize(
            comnpute_expectation_tn,
            init,
            args=(estimator,),
            method="COBYLA",
            options={
                "maxiter": 100
            },
        )

        ansatz = QAOAAnsatz(
            cost_operator=qubit_op,
            reps=n_reps,
            initial_state=TestQAOA.make_initial_state_circuit(),
            mixer_operator=mixer,
            name='QAOA',
            flatten=None,
        )
        mapping = operands["make_pname2theta"](result.x)
        parameter2value = {param: mapping[param.name] for param in ansatz.parameters}
        opt_ansatz = ansatz.bind_parameters(parameter2value)
        opt_ansatz.measure_all()

        sim = AerSimulator(device="GPU", method="tensor_network")
        t_qc = transpile(opt_ansatz, backend=sim)
        shots = 1024
        counts = sim.run(t_qc, shots=shots).result().get_counts()
        state, _ = sorted(counts.items(), key=lambda k_v: -k_v[1])[0]
        self.assertEqual(state, answer)
