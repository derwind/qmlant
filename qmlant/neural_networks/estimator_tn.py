from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import cupy as cp
import numpy as np
from cuquantum import contract

from .utils import Pauli, replace_by_batch, replace_pauli, replace_pauli_phase_shift


def default_make_pname2theta(params: Sequence[float] | np.ndarray) -> dict[str, float]:
    return {f"θ[{i}]": params[i] for i in range(len(params))}


def default_batch_filter(params: np.ndarray) -> np.ndarray:
    return params  # do nothing


class EstimatorTN:
    """A neural network implementation based on the cuTensorNet.

    Args:
        pname2locs (dict[str, tuple[int, int]]): a dict whose keys are f"θ[{i}]" and values are locations (loc, dag_loc) for Ry(θ±π/2) and Ry(-(θ±π/2)) respectively in `operands`
        pname_symbol (str): the symbol for ansatz portion (default: "θ")
        expr (str | None): initial `expr` by `CircuitToEinsum`
        operands (list[cp.ndarray] | None): initial `operands` by `CircuitToEinsum`
        make_pname2theta (Callable): a function which generates `pname2theta`, e.g. `{"x[0]": 0.23, "x[1]": -0.54, "θ[0]": 1.32}`
        batch_filter (Callable): a function which converts batch data, e.g. `lambda batch: batch * 2`
    """

    def __init__(
        self,
        pname2locs: dict[str, tuple[list[int], list[int], Pauli]],
        expr: str | None = None,
        operands: list[cp.ndarray] | None = None,
        make_pname2theta: Callable[
            [Sequence[float] | np.ndarray], dict[str, float]
        ] = default_make_pname2theta,
        batch_filter: Callable[[np.ndarray], np.ndarray] = default_batch_filter,
    ):
        self.pname2locs = pname2locs
        self.expr = expr
        self.operands = operands
        self.make_pname2theta = make_pname2theta
        self.batch_filter = batch_filter
        self._params: Sequence[float] | np.ndarray | None = None  # cache params of forward
        self._last_forward: np.ndarray | None = None  # cache expvals of forward

    def _prepare_circuit(
        self,
        params: Sequence[float] | np.ndarray,
        expr: str,
        operands: list[cp.ndarray],
        batch: np.ndarray | None,
    ) -> tuple[str, list[cp.ndarray]]:
        """prepare a circuit for forward process setting batch and parameters to the circuit."""

        if batch is not None:
            pname2theta_list = {
                f"x[{i}]": self.batch_filter(batch[:, i]).flatten().tolist()
                for i in range(batch.shape[1])
            }

            expr, operands = replace_by_batch(expr, operands, pname2theta_list, self.pname2locs)

        pname2theta = self.make_pname2theta(params)
        operands = replace_pauli(operands, pname2theta, self.pname2locs)

        return expr, operands

    def forward(
        self,
        params: Sequence[float] | np.ndarray,
        batch: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward pass of the network.

        Args:
            params (Sequence[float] | np.ndarray): parameters for ansatz
            batch (np.ndarray | None): batch data

        Returns:
            expectation values
        """

        self.expr, self.operands = self._prepare_circuit(params, self.expr, self.operands, batch)
        self._params = params
        self._last_forward = cp.asnumpy(contract(self.expr, *self.operands).real.reshape(-1, 1))
        if batch is None:
            self._last_forward = np.sum(self._last_forward)
        return self._last_forward

    def forward_with_tn(
        self,
        params: Sequence[float] | np.ndarray,
        expr: str,
        operands: list[cp.ndarray],
        batch: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, list[cp.ndarray]]:
        """Forward pass of the network with a Tensor Network (expr and operands).

        Args:
            params (Sequence[float] | np.ndarray): parameters for ansatz
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`
            batch (np.ndarray | None): batch data

        Returns:
            expectation values, possible updated expr and possible updated operands
        """

        expr, operands = self._prepare_circuit(params, expr, operands, batch)
        self._params = params
        self._last_forward = cp.asnumpy(contract(expr, *operands).real.reshape(-1, 1))
        if batch is None:
            self._last_forward = np.sum(self._last_forward)
        return self._last_forward, expr, operands

    def _check_expr_and_operands(
        self, expr: str | None = None, operands: list[cp.ndarray] | None = None
    ):
        if expr is None:
            expr = self.expr
        if operands is None:
            operands = self.operands

        if expr is None:
            raise ValueError("`expr` must not be None.")
        if operands is None:
            raise ValueError("`operands` must not be None.")

        return expr, operands

    def pop_last_forward(self) -> np.ndarray:
        last_forward = self._last_forward
        self._last_forward = None
        return last_forward

    def _prepare_backward_circuit(
        self, forward_expr: str, forward_operands: list[cp.ndarray], param_symbol: str = "孕"
    ) -> tuple[str, list[cp.ndarray]]:
        # ansatz portion only
        pname2locs: dict[str, set[int]] = {
            pname: set(locs + dag_locs)
            for pname, (locs, dag_locs, _) in self.pname2locs.items()
            if not pname.startswith("x")  # "θ[i]" etc.
        }
        n_params = len(pname2locs)
        param_locs: set[int] = set()
        for locs in pname2locs.values():
            param_locs.update(locs)

        ins, out = re.split(r"\s*->\s*", forward_expr)
        ins = re.split(r"\s*,\s*", ins)

        backward_ins = []
        backward_operands = []

        for i, (idx, ops) in enumerate(zip(ins, forward_operands)):
            if i not in param_locs:  # not ansatz params
                backward_ins.append(idx)
                backward_operands.append(ops)
                continue

            # multiplex ansatz params for simultaneous parameter-shift calculations
            backward_ins.append(param_symbol + idx)
            ops_ = cp.ascontiguousarray(cp.broadcast_to(ops, (2 * n_params, *ops.shape)))
            backward_operands.append(ops_)

        return ",".join(backward_ins) + "->" + out + param_symbol, backward_operands

    def backward(
        self,
        expr: str | None = None,
        operands: list[cp.ndarray] | None = None,
    ) -> np.ndarray:
        """Backward pass of the network.

        Args:
            params (Sequence[float]): phase params for ansatz portion
            expr (str | None): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray] | None): `operands` by `CircuitToEinsum`

        Returns:
            gradient values
        """

        if self._params is None:
            raise RuntimeError("call forward before backward")

        expr, operands = self._check_expr_and_operands(expr, operands)
        expr, operands = self._prepare_backward_circuit(expr, operands)

        pname2theta = self.make_pname2theta(self._params)
        self._params = None

        operands = replace_pauli_phase_shift(operands, pname2theta, self.pname2locs)
        #     p0_p, p0_m, p1_p, p1_m, ...
        # b0  xx    xx    xx    xx
        # b1  xx    xx    xx    xx
        # b2  xx    xx    xx    xx
        expvals = cp.asnumpy(contract(expr, *operands).real)
        for i in range(0, expvals.shape[1], 2):
            expvals[:, i] = (expvals[:, i] - expvals[:, i + 1]) / 2
        return expvals[:, range(0, expvals.shape[1], 2)]
