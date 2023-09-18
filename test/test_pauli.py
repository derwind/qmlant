import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import math
import time
from collections.abc import Sequence

import cupy as cp
import numpy as np

from qmlant.models.binary_classification import TTN
from qmlant.neural_networks.utils import (
    Rx_Rxdag,
    Rxx_Rxxdag,
    Ry_Rydag,
    Ryy_Ryydag,
    Rz_Rzdag,
    Rzz_Rzzdag,
)
from qmlant.transforms import MapLabel, ToTensor

class TestPauli(unittest.TestCase):
    def test_Rx_Rxdag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Rx_Rxdag(angle)
            mat = cp.zeros((2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2), dtype=complex)
            Rx_Rxdag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))

    def test_Ry_Rydag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Ry_Rydag(angle)
            mat = cp.zeros((2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2), dtype=complex)
            Ry_Rydag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))

    def test_Rz_Rzdag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Rz_Rzdag(angle)
            mat = cp.zeros((2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2), dtype=complex)
            Rz_Rzdag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))

    def test_Rxx_Rxxdag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Rxx_Rxxdag(angle)
            mat = cp.zeros((2, 2, 2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2, 2, 2), dtype=complex)
            Rxx_Rxxdag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))

    def test_Ryy_Ryydag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Ryy_Ryydag(angle)
            mat = cp.zeros((2, 2, 2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2, 2, 2), dtype=complex)
            Ryy_Ryydag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))

    def test_Rzz_Rzzdag(self):
        for angle in np.arange(0, np.pi * 2, 0.01):
            answer_mat, answer_mat_dag = Rzz_Rzzdag(angle)
            mat = cp.zeros((2, 2, 2, 2), dtype=complex)
            mat_dag = cp.zeros((2, 2, 2, 2), dtype=complex)
            Rzz_Rzzdag(angle, mat, mat_dag)
            self.assertTrue(cp.all(mat == answer_mat))
            self.assertTrue(cp.all(mat_dag == answer_mat_dag))
