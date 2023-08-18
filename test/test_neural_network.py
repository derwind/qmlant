import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import cupy as cp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from qiskit import QuantumCircuit  # pylint: disable=wrong-import-position
from qiskit.circuit import ParameterVector  # pylint: disable=wrong-import-position

from qmlant.circuit.library import ZFeatureMap  # pylint: disable=wrong-import-position
from qmlant.neural_networks import (  # pylint: disable=wrong-import-position
    Ry,
    Ry_Rydag,
    Rz_Rzdag,
    circuit_to_einsum_expectation,
    replace_by_batch,
    replace_pauli,
    replace_pauli_phase_shift,
)


class TestRy(unittest.TestCase):
    @classmethod
    def naive_Ry(cls, theta):
        return cp.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

    def test_Ry(self):
        mat = Ry(np.pi / 3)
        answer = self.naive_Ry(np.pi / 3)
        self.assertTrue(cp.all(mat == answer))

    def test_Ry_Rydag(self):
        mat, mat_dag = Ry_Rydag(np.pi / 3)
        answer = self.naive_Ry(np.pi / 3)
        answer_dag = self.naive_Ry(-np.pi / 3)
        self.assertTrue(cp.all(mat == answer) and cp.all(mat_dag == answer_dag))


class TestReplacer(unittest.TestCase):
    ZERO = cp.array([1, 0], dtype=complex)
    X = cp.array([[0, 1], [1, 0]], dtype=complex)
    Y = cp.array([[0, -1.0j], [1.0j, 0]], dtype=complex)
    Z = cp.array([[1, 0], [0, -1]], dtype=complex)
    H = cp.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    DUMMY = cp.array([[0, 0], [0, 0]], dtype=complex)

    @classmethod
    def naive_Ry(cls, theta):
        return cp.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

    def test_circuit_to_einsum_expectation(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        answer_expr = "a,b,ca,dc,eb,fe,gd,hf,ih,ji,kg,lk,l,j->"
        answer_operands = [
            self.ZERO,
            self.ZERO,
            self.X,
            self.DUMMY,
            self.X,
            self.DUMMY,
            self.Z,
            self.Z,
            self.DUMMY,
            self.X,
            self.DUMMY,
            self.X,
            self.ZERO,
            self.ZERO,
        ]
        answer_params2locs = {"x[0]": ([3], [10], Ry_Rydag), "x[1]": ([5], [8], Ry_Rydag)}

        expr, operands, params2locs = circuit_to_einsum_expectation(qc, "ZZ")
        self.assertEqual(expr, answer_expr)
        self.assertEqual(len(operands), len(answer_operands))
        for ops, ops_ans in zip(operands, answer_operands):
            if cp.all(ops_ans == self.DUMMY):
                continue
            self.assertTrue(cp.all(ops == ops_ans))
        self.assertEqual(params2locs, answer_params2locs)

    def test_circuit_to_einsum_expectation2(self):
        qc = ZFeatureMap(3, parameter_multiplier=1)

        answer_expr = "a,b,c,da,eb,fc,gd,he,if,jg,kh,li,mj,nk,ol,pm,qn,ro,sr,tq,up,vs,wt,xu,yv,zw,Ax,By,Cz,DA,D,C,B->"
        answer_operands = [
            self.ZERO,
            self.ZERO,
            self.ZERO,
            self.H,
            self.H,
            self.H,
            self.DUMMY,
            self.DUMMY,
            self.DUMMY,
            self.H,
            self.H,
            self.H,
            self.DUMMY,
            self.DUMMY,
            self.DUMMY,
            self.Z,
            self.Z,
            self.Z,
            self.DUMMY,
            self.DUMMY,
            self.DUMMY,
            self.H,
            self.H,
            self.H,
            self.DUMMY,
            self.DUMMY,
            self.DUMMY,
            self.H,
            self.H,
            self.H,
            self.ZERO,
            self.ZERO,
            self.ZERO,
        ]
        answer_params2locs = {
            "x[0]": ([6, 12], [26, 20], Rz_Rzdag),
            "x[1]": ([7, 13], [25, 19], Rz_Rzdag),
            "x[2]": ([8, 14], [24, 18], Rz_Rzdag),
        }

        expr, operands, params2locs = circuit_to_einsum_expectation(qc, "ZZZ")
        self.assertEqual(expr, answer_expr)
        self.assertEqual(len(operands), len(answer_operands))
        for ops, ops_ans in zip(operands, answer_operands):
            if cp.all(ops_ans == self.DUMMY):
                continue
            self.assertTrue(cp.all(ops == ops_ans))
        self.assertEqual(params2locs, answer_params2locs)

    def test_replace_by_batch(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        answer_expr = "a,b,ca,dc,eb,fe,gd,hf,ih,ji,kg,lk,l,j->"
        answer_pname2locs = {"x[0]": ([3], [10], Ry_Rydag), "x[1]": ([5], [8], Ry_Rydag)}
        expr, operands, pname2locs = circuit_to_einsum_expectation(qc, "ZZ")
        self.assertEqual(expr, answer_expr)
        self.assertEqual(pname2locs, answer_pname2locs)

        pname2theta_list = {"x[0]": [np.pi / 6, np.pi / 3], "x[1]": [np.pi / 4, np.pi / 8]}
        expr2, operands2 = replace_by_batch(
            expr, operands, pname2theta_list, pname2locs, batch_symbol="ξ"
        )
        answer_expr2 = "a,b,ca,ξdc,eb,ξfe,gd,hf,ξih,ji,ξkg,lk,l,j->ξ"
        answer_operands2 = [
            self.ZERO,
            self.ZERO,
            self.X,
            cp.array([self.naive_Ry(np.pi / 6), self.naive_Ry(np.pi / 3)]),
            self.X,
            cp.array([self.naive_Ry(np.pi / 4), self.naive_Ry(np.pi / 8)]),
            self.Z,
            self.Z,
            cp.array([self.naive_Ry(-np.pi / 4), self.naive_Ry(-np.pi / 8)]),
            self.X,
            cp.array([self.naive_Ry(-np.pi / 6), self.naive_Ry(-np.pi / 3)]),
            self.X,
            self.ZERO,
            self.ZERO,
        ]

        self.assertEqual(expr2, answer_expr2)
        for ops, ops_ans in zip(operands2, answer_operands2):
            self.assertTrue(cp.all(ops == ops_ans))

    def test_replace_pauli(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        answer_expr = "a,b,ca,dc,eb,fe,gd,hf,ih,ji,kg,lk,l,j->"
        answer_pname2locs = {"x[0]": ([3], [10], Ry_Rydag), "x[1]": ([5], [8], Ry_Rydag)}
        expr, operands, pname2locs = circuit_to_einsum_expectation(qc, "ZZ")
        self.assertEqual(expr, answer_expr)

        pname2theta_list = {"x[0]": np.pi / 6, "x[1]": np.pi / 4}
        operands2 = replace_pauli(operands, pname2theta_list, pname2locs)
        answer_operands2 = [
            self.ZERO,
            self.ZERO,
            self.X,
            cp.array([self.naive_Ry(np.pi / 6)]),
            self.X,
            cp.array([self.naive_Ry(np.pi / 4)]),
            self.Z,
            self.Z,
            cp.array([self.naive_Ry(-np.pi / 4)]),
            self.X,
            cp.array([self.naive_Ry(-np.pi / 6)]),
            self.X,
            self.ZERO,
            self.ZERO,
        ]
        self.assertEqual(pname2locs, answer_pname2locs)

        for ops, ops_ans in zip(operands2, answer_operands2):
            self.assertTrue(cp.all(ops == ops_ans))

    def test_replace_pauli_phase_shift(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        expr, operands, pname2locs = circuit_to_einsum_expectation(qc, "ZZ")

        pname2theta_list = {"x[0]": [np.pi / 6] * 4, "x[1]": [np.pi / 4] * 4}
        _, operands2 = replace_by_batch(
            expr, operands, pname2theta_list, pname2locs, batch_symbol="ξ"
        )

        pname2theta = {"x[0]": np.pi / 6, "x[1]": np.pi / 4}
        operands3 = replace_pauli_phase_shift(operands2, pname2theta, pname2locs)

        answer_operands3 = [
            self.ZERO,
            self.ZERO,
            self.X,
            cp.array(
                [
                    self.naive_Ry(np.pi / 6 + np.pi / 2),
                    self.naive_Ry(np.pi / 6 - np.pi / 2),
                    self.naive_Ry(np.pi / 6),
                    self.naive_Ry(np.pi / 6),
                ]
            ),
            self.X,
            cp.array(
                [
                    self.naive_Ry(np.pi / 4),
                    self.naive_Ry(np.pi / 4),
                    self.naive_Ry(np.pi / 4 + np.pi / 2),
                    self.naive_Ry(np.pi / 4 - np.pi / 2),
                ]
            ),
            self.Z,
            self.Z,
            cp.array(
                [
                    self.naive_Ry(-np.pi / 4),
                    self.naive_Ry(-np.pi / 4),
                    self.naive_Ry(-(np.pi / 4 + np.pi / 2)),
                    self.naive_Ry(-(np.pi / 4 - np.pi / 2)),
                ]
            ),
            self.X,
            cp.array(
                [
                    self.naive_Ry(-(np.pi / 6 + np.pi / 2)),
                    self.naive_Ry(-(np.pi / 6 - np.pi / 2)),
                    self.naive_Ry(-np.pi / 6),
                    self.naive_Ry(-np.pi / 6),
                ]
            ),
            self.X,
            self.ZERO,
            self.ZERO,
        ]

        for ops, ops_ans in zip(operands3, answer_operands3):
            self.assertTrue(cp.all(ops == ops_ans))
