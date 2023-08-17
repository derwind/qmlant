import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import math  # pylint: disable=wrong-import-position
import time  # pylint: disable=wrong-import-position
from collections.abc import Sequence  # pylint: disable=wrong-import-position

import cupy as cp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import torch  # pylint: disable=wrong-import-position
from torchvision import transforms  # pylint: disable=wrong-import-position
from cuquantum import contract  # pylint: disable=wrong-import-position
from qiskit import QuantumCircuit  # pylint: disable=wrong-import-position
from torch.utils.data import DataLoader, Dataset  # pylint: disable=wrong-import-position

from qmlant import optim  # pylint: disable=wrong-import-position
from qmlant.datasets import Iris  # pylint: disable=wrong-import-position
from qmlant.models.binary_classification import TTN  # pylint: disable=wrong-import-position
from qmlant.neural_networks import EstimatorTN  # pylint: disable=wrong-import-position
from qmlant.neural_networks.utils import (  # pylint: disable=wrong-import-position
    find_pauli_locs,
    replace_by_batch,
    replace_pauli,
)
from qmlant.transforms import MapLabel, ToTensor  # pylint: disable=wrong-import-position


class PQCTrainerTN:  # pylint: disable=too-few-public-methods
    def __init__(
        self, qc: QuantumCircuit, initial_point: Sequence[float], optimizer: optim.Optimizer
    ):
        self.qc_pl = qc  # placeholder circuit
        self.initial_point = np.array(initial_point)
        self.optimizer = optimizer

    def fit(
        self,
        dataset: Dataset,
        batch_size: int,
        operator: str,
        callbacks: list | None = None,
        epochs: int = 1,
    ) -> None:
        pname2locs, expr, oprands = find_pauli_locs(self.qc_pl, operator, return_tn=True)

        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        callbacks = callbacks if callbacks is not None else []

        params = self.initial_point.copy()
        if isinstance(params, list):
            params = np.array(params)

        qnn = EstimatorTN(pname2locs)

        for _ in range(epochs):
            for batch, label in dataloader:
                batch, label = self._preprocess_batch(batch, label)
                label = label.reshape(label.shape[0], -1)

                expr, oprands = qnn.prepare_circuit(batch, params, expr, oprands)

                # "forward"
                expvals = qnn.forward(expr, oprands)
                total_loss = np.mean((expvals - label) ** 2)

                # "backward"
                # The parameter-shift rule
                # [[∂f1/∂θ1, ∂f1/∂θ2, ..., ∂f1/∂θn],
                #  [∂f2/∂θ1, ∂f2/∂θ2, ..., ∂f2/∂θn],
                #  ...]
                grads = qnn.backward(expr, oprands, params)
                expvals_minus_label = (expvals - label).reshape(batch.shape[0], -1)
                total_grads = np.mean(expvals_minus_label * grads, axis=0)

                for callback in callbacks:
                    callback(total_loss, params)

                # "update params"
                self.optimizer.update(params, total_grads)

    def _preprocess_batch(
        self, batch: torch.Tensor, label: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_: np.ndarray = batch.detach().numpy()
        label_: np.ndarray = label.detach().numpy()
        return batch_, label_


def RunPQCTrain(
    dataset: Dataset,
    batch_size: int,
    qc: QuantumCircuit,
    operator: str,
    init: Sequence[float] | None = None,
    epochs: int = 1,
    interval: int = 100,  # pylint: disable=unused-argument
) -> tuple[list[float], list[float]]:
    opt_params = None
    opt_loss = None

    def save_opt_params(loss, params):
        nonlocal opt_params, opt_loss

        if opt_loss is None or loss < opt_loss:
            opt_params = params.copy()
            opt_loss = loss

    # Store intermediate results
    history: dict[str, list[float]] = {"loss": [], "params": []}

    def store_intermediate_result(loss, params):  # pylint: disable=unused-argument
        history["loss"].append(loss)
        history["params"].append(None)

    optimizer = optim.Adam(alpha=0.01)
    trainer = PQCTrainerTN(qc=qc, initial_point=init, optimizer=optimizer)
    trainer.fit(
        dataset,
        batch_size,
        operator,
        callbacks=[save_opt_params, store_intermediate_result],
        epochs=epochs,
    )

    return opt_params.tolist(), history["loss"]


class TestIrus(unittest.TestCase):
    def test_iris_classification(self):
        target_transform = transforms.Compose([MapLabel([1, 2], [1, -1]), ToTensor(int)])

        trainset = Iris(
            test_size=0.3,
            transform=ToTensor(),
            target_transform=target_transform,
            subclass_targets=[1, 2],
        )

        testset = Iris(
            train=False,
            test_size=0.3,
            transform=ToTensor(),
            target_transform=target_transform,
            subclass_targets=[1, 2],
        )

        n_qubits = 4

        x_length, length = TTN.make_placeholder_circuit(n_qubits, dry_run=True)
        self.assertEqual(x_length, n_qubits)
        self.assertEqual(length, 7)

        time_start = time.time()

        placeholder_circuit = TTN.make_placeholder_circuit(n_qubits)
        hamiltonian = TTN.get_hamiltonian(n_qubits)

        np.random.seed(10)
        init = np.random.random(length) * 2 * math.pi

        opt_params, loss_list = RunPQCTrain(
            trainset, 64, placeholder_circuit, hamiltonian, init=init, epochs=100
        )

        elapsed_time = time.time() - time_start
        self.assertLess(elapsed_time, 20)

        min_loss = min(loss_list)
        self.assertLess(min_loss, 0.3)
        self.assertTrue(3.5 < opt_params[0] < 4.5, opt_params[0])
        self.assertTrue(-0.6 < opt_params[1] < -0.5, opt_params[1])
        self.assertTrue(4.5 < opt_params[2] < 5.5, opt_params[2])
        self.assertTrue(5.0 < opt_params[3] < 6.0, opt_params[3])
        self.assertTrue(2.0 < opt_params[4] < 3.0, opt_params[4])
        self.assertTrue(1.5 < opt_params[5] < 2.5, opt_params[5])
        self.assertTrue(0.5 < opt_params[6] < 1.5, opt_params[6])

        params = opt_params
        pname2locs, expr, operands = find_pauli_locs(placeholder_circuit, hamiltonian, return_tn=True)
        pname2theta = {f"θ[{i}]": params[i] for i in range(len(params))}

        testloader = DataLoader(testset, 32)

        total = 0
        total_correct = 0

        for batch, label in testloader:
            batch, label = batch.detach().numpy(), label.detach().numpy()

            pname2theta_list = {
                f"x[{i}]": batch[:, i].flatten().tolist() for i in range(batch.shape[1])
            }
            expr, operands = replace_by_batch(expr, operands, pname2theta_list, pname2locs)

            operands = replace_pauli(operands, pname2theta, pname2locs)

            expvals = cp.asnumpy(contract(expr, *operands).real)

            predict_labels = np.ones_like(expvals)
            predict_labels[np.where(expvals < 0)] = -1
            predict_labels = predict_labels.astype(int)

            total_correct += np.sum(predict_labels == label)
            total += batch.shape[0]

        acc = total_correct / total
        self.assertGreater(acc, 0.85)
