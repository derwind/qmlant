from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import cupy as cp
from cuquantum import contract


class EstimatorTN:
    """A neural network implementation based on the cuTensorNet.

    Args:
        pname2locs (dict[str, tuple[int, int]]): a dict whose keys are f"θ[{i}]" and values are locations (loc, dag_loc) for Ry(θ±π/2) and Ry(-(θ±π/2)) respectively in `operands`
        pname_symbol (str): the symbol for ansatz portion (default: "θ")
    """

    def __init__(self,
        pname2locs: dict[str, tuple[int, int]],
        pname_symbol: str = "θ"
    ):
        self.pname2locs = pname2locs
        self.pname_symbol = pname_symbol

    def forward(self,
        expr: str,
        operands: list[cp.ndarray]
    ) -> np.ndarray:
        """Forward pass of the network.

        Args:
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`

        Returns:
            expectation values
        """

        return cp.asnumpy(contract(expr, *operands).real.reshape(-1, 1))

    def backward(self,
        expr: str,
        operands: list[cp.ndarray],
        params: Sequence[float]
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
                operands, pname, pname2theta, self.pname2locs, np.pi/2
            ):
                expvals_p = cp.asnumpy(contract(expr, *operands).real.flatten())
            with self.temporarily_replace_ry(
                operands, pname, pname2theta, self.pname2locs, -np.pi/2
            ):
                expvals_m = cp.asnumpy(contract(expr, *operands).real.flatten())
            expvals = ((expvals_p - expvals_m) / 2)
            expval_array.append(expvals)

        # batch grads_i is converted to a column vector
        return np.array(expval_array).T
