import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import cupy as cp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from qiskit import QuantumCircuit  # pylint: disable=wrong-import-position
from qiskit.circuit import ParameterVector  # pylint: disable=wrong-import-position

from qmlant.neural_networks import Ry, Ry_Rydag  # pylint: disable=wrong-import-position
from qmlant.neural_networks.utils import (  # pylint: disable=wrong-import-position
    find_ry_locs,
    replace_by_batch,
    replace_ry,
    replace_ry_phase_shift,
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
        self.assertEqual(cp.all(mat == answer), True)

    def test_Ry_Rydag(self):
        mat, mat_dag = Ry_Rydag(np.pi / 3)
        answer = self.naive_Ry(np.pi / 3)
        answer_dag = self.naive_Ry(-np.pi / 3)
        self.assertEqual(cp.all(mat == answer) and cp.all(mat_dag == answer_dag), True)


class TestReplacer(unittest.TestCase):
    ZERO = cp.array([1, 0], dtype=complex)
    X = cp.array([[0, 1], [1, 0]], dtype=complex)
    Y = cp.array([[0, -1.0j], [1.0j, 0]], dtype=complex)
    Z = cp.array([[1, 0], [0, -1]], dtype=complex)
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

    def test_find_ry_locs(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        params2locs = find_ry_locs(qc, "ZZ", return_tn=False)
        answer = {"x[0]": (3, 10), "x[1]": (5, 8)}
        self.assertEqual(params2locs, answer)

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

        params2locs, expr, operands = find_ry_locs(qc, "ZZ", return_tn=True)
        self.assertEqual(params2locs, answer)
        self.assertEqual(expr, answer_expr)
        self.assertEqual(len(operands), len(answer_operands))
        for ops, ops_ans in zip(operands, answer_operands):
            if cp.all(ops_ans == self.DUMMY):
                continue
            self.assertEqual(cp.all(ops == ops_ans), True)

    def test_replace_by_batch(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        pname2locs = find_ry_locs(qc, "ZZ", return_tn=False)
        answer = {"x[0]": (3, 10), "x[1]": (5, 8)}
        self.assertEqual(pname2locs, answer)

        answer_expr = "a,b,ca,dc,eb,fe,gd,hf,ih,ji,kg,lk,l,j->"
        pname2locs, expr, operands = find_ry_locs(qc, "ZZ", return_tn=True)
        self.assertEqual(expr, answer_expr)

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
            self.assertEqual(cp.all(ops == ops_ans), True)

    def test_replace_ry(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        pname2locs = find_ry_locs(qc, "ZZ", return_tn=False)
        answer = {"x[0]": (3, 10), "x[1]": (5, 8)}
        self.assertEqual(pname2locs, answer)

        answer_expr = "a,b,ca,dc,eb,fe,gd,hf,ih,ji,kg,lk,l,j->"
        pname2locs, expr, operands = find_ry_locs(qc, "ZZ", return_tn=True)
        self.assertEqual(expr, answer_expr)

        pname2theta_list = {"x[0]": np.pi / 6, "x[1]": np.pi / 4}
        operands2 = replace_ry(operands, pname2theta_list, pname2locs)
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

        for ops, ops_ans in zip(operands2, answer_operands2):
            self.assertEqual(cp.all(ops == ops_ans), True)

    def test_replace_ry_phase_shift(self):
        params = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)

        pname2locs, expr, operands = find_ry_locs(qc, "ZZ", return_tn=True)

        pname2theta_list = {"x[0]": [np.pi / 6] * 4, "x[1]": [np.pi / 4] * 4}
        _, operands2 = replace_by_batch(
            expr, operands, pname2theta_list, pname2locs, batch_symbol="ξ"
        )

        pname2theta = {"x[0]": np.pi / 6, "x[1]": np.pi / 4}
        operands3 = replace_ry_phase_shift(operands2, pname2theta, pname2locs)

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
            self.assertEqual(cp.all(ops == ops_ans), True)
