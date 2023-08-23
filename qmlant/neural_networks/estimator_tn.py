from __future__ import annotations

import contextlib
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

    def _prepare_circuit(
        self,
        batch: np.ndarray,
        params: Sequence[float] | np.ndarray,
        expr: str,
        operands: list[cp.ndarray],
    ) -> tuple[str, list[cp.ndarray]]:
        """prepare a circuit for forward process setting batch and parameters to the circuit."""

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
        batch: np.ndarray,
        params: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Forward pass of the network.

        Args:
            batch (np.ndarray): batch data
            params (Sequence[float] | np.ndarray): parameters for ansatz

        Returns:
            expectation values
        """

        self.expr, self.operands = self._prepare_circuit(batch, params, self.expr, self.operands)
        self._params = params
        return cp.asnumpy(contract(self.expr, *self.operands).real.reshape(-1, 1))

    def forward_with_tn(
        self,
        batch: np.ndarray,
        params: Sequence[float] | np.ndarray,
        expr: str,
        operands: list[cp.ndarray],
    ) -> tuple[np.ndarray, str, list[cp.ndarray]]:
        """Forward pass of the network with a Tensor Network (expr and operands).

        Args:
            batch (np.ndarray): batch data
            params (Sequence[float] | np.ndarray): parameters for ansatz
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`

        Returns:
            expectation values, possible updated expr and possible updated operands
        """

        expr, operands = self._prepare_circuit(batch, params, expr, operands)
        self._params = params
        return cp.asnumpy(contract(expr, *operands).real.reshape(-1, 1)), expr, operands

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
            ops = cp.ascontiguousarray(cp.broadcast_to(ops, (2 * n_params, *ops.shape)))
            backward_operands.append(ops)

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

    def old_backward(
        self,
        params: Sequence[float] | np.ndarray,
        expr: str | None = None,
        operands: list[cp.ndarray] | None = None,
    ) -> np.ndarray:
        """Backward pass of the network.

        Args:
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`
            params (Sequence[float]): phase params for ansatz portion

        Returns:
            gradient values
        """

        pname2theta = self.make_pname2theta(params)

        expval_array = []
        for i in range(len(pname2theta)):
            pname = f"θ[{i}]"
            # per batch
            with self.temporarily_replace_pauli(
                operands, pname, pname2theta, self.pname2locs, np.pi / 2
            ):
                expvals_p = cp.asnumpy(contract(expr, *operands).real.flatten())
            with self.temporarily_replace_pauli(
                operands, pname, pname2theta, self.pname2locs, -np.pi / 2
            ):
                expvals_m = cp.asnumpy(contract(expr, *operands).real.flatten())
            expvals = (expvals_p - expvals_m) / 2
            expval_array.append(expvals)

        # batch grads_i is converted to a column vector
        return np.array(expval_array).T

    @classmethod
    @contextlib.contextmanager
    def temporarily_replace_pauli(
        cls,
        operands: list[cp.ndarray],
        pname: str,
        pname2theta: dict[str, float],
        pname2locs: dict[str, tuple[list[int], list[int], Pauli]],
        phase_shift: float = np.pi / 2,
    ):
        backups = {}
        try:
            theta = pname2theta[pname]  # e.g. pname = "θ[0]"
            locs, dag_locs, make_paulis = pname2locs[pname]
            for loc, dag_loc in zip(locs, dag_locs):
                backups.update({loc: operands[loc], dag_loc: operands[dag_loc]})
                operands[loc], operands[dag_loc] = make_paulis(theta + phase_shift)
            yield operands
        finally:
            for i, v in backups.items():
                operands[i] = v
