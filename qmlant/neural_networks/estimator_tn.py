from __future__ import annotations

import contextlib
from collections.abc import Sequence

import cupy as cp
import numpy as np
from cuquantum import contract

from .neural_network import Ry


class EstimatorTN:
    """A neural network implementation based on the cuTensorNet.

    Args:
        pname2locs (dict[str, tuple[int, int]]): a dict whose keys are f"θ[{i}]" and values are locations (loc, dag_loc) for Ry(θ±π/2) and Ry(-(θ±π/2)) respectively in `operands`
        pname_symbol (str): the symbol for ansatz portion (default: "θ")
    """

    def __init__(self, pname2locs: dict[str, tuple[int, int]], pname_symbol: str = "θ"):
        self.pname2locs = pname2locs
        self.pname_symbol = pname_symbol

    def forward(self, expr: str, operands: list[cp.ndarray]) -> np.ndarray:
        """Forward pass of the network.

        Args:
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`

        Returns:
            expectation values
        """

        return cp.asnumpy(contract(expr, *operands).real.reshape(-1, 1))

    def backward(
        self, expr: str, operands: list[cp.ndarray], params: Sequence[float]
    ) -> np.ndarray:
        """Backward pass of the network.

        Args:
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`
            params (Sequence[float]): phase params for ansatz portion

        Returns:
            gradient values
        """

        pname2theta = {f"{self.pname_symbol}[{i}]": params[i] for i in range(len(params))}

        expval_array = []
        for i in range(len(pname2theta)):
            pname = f"θ[{i}]"
            # per batch
            with self.temporarily_replace_ry(
                operands, pname, pname2theta, self.pname2locs, np.pi / 2
            ):
                expvals_p = cp.asnumpy(contract(expr, *operands).real.flatten())
            with self.temporarily_replace_ry(
                operands, pname, pname2theta, self.pname2locs, -np.pi / 2
            ):
                expvals_m = cp.asnumpy(contract(expr, *operands).real.flatten())
            expvals = (expvals_p - expvals_m) / 2
            expval_array.append(expvals)

        # batch grads_i is converted to a column vector
        return np.array(expval_array).T

    @classmethod
    @contextlib.contextmanager
    def temporarily_replace_ry(
        cls,
        operands: list[cp.ndarray],
        pname: str,
        pname2theta: dict[str, float],
        pname2locs: dict[str, tuple[int, int]],
        phase_shift: float = np.pi / 2,
    ):
        backups = {}
        try:
            theta = pname2theta[pname]  # e.g. pname = "θ[0]"
            loc, dag_loc = pname2locs[pname]
            backups = {loc: operands[loc], dag_loc: operands[dag_loc]}
            operands[loc] = Ry(theta + phase_shift)
            operands[dag_loc] = Ry(-(theta + phase_shift))
            yield operands
        finally:
            for i, v in backups.items():
                operands[i] = v
