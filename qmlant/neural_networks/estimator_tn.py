from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence

import cupy as cp
import numpy as np
from cuquantum import contract

from .utils import Pauli, replace_by_batch, replace_pauli, replace_pauli_phase_shift


class EstimatorTN:
    """A neural network implementation based on the cuTensorNet.

    Args:
        pname2locs (dict[str, tuple[int, int]]): a dict whose keys are f"θ[{i}]" and values are locations (loc, dag_loc) for Ry(θ±π/2) and Ry(-(θ±π/2)) respectively in `operands`
        pname_symbol (str): the symbol for ansatz portion (default: "θ")
    """

    def __init__(self, pname2locs: dict[str, tuple[int, int, Pauli]], pname_symbol: str = "θ"):
        self.pname2locs = pname2locs
        self.pname_symbol = pname_symbol

    def prepare_circuit(
        self,
        batch: np.ndarray,
        params: Sequence[float] | np.ndarray,
        expr: str,
        operands: list[cp.ndarray],
    ) -> tuple[str, list[cp.ndarray]]:
        """prepare a circuit for forward process setting batch and parameters to the circuit."""

        pname2theta_list = {
            f"x[{i}]": batch[:, i].flatten().tolist() for i in range(batch.shape[1])
        }
        expr, operands = replace_by_batch(expr, operands, pname2theta_list, self.pname2locs)

        pname2theta = {f"{self.pname_symbol}[{i}]": params[i] for i in range(len(params))}
        operands = replace_pauli(operands, pname2theta, self.pname2locs)

        return expr, operands

    def _prepare_backward_circuit(
        self, forward_expr: str, forward_operands: list[cp.ndarray], param_symbol: str = "孕"
    ) -> tuple[str, list[cp.ndarray]]:
        # ansatz portion only
        pname2locs = {
            pname: locs
            for pname, locs in self.pname2locs.items()
            if pname.startswith(self.pname_symbol)
        }
        n_params = len(pname2locs)
        param_locs = set()
        for loc, loc_dag, _ in pname2locs.values():
            param_locs.add(loc)
            param_locs.add(loc_dag)

        ins, out = re.split(r"\s*->\s*", forward_expr)
        ins = re.split(r"\s*,\s*", ins)

        backward_ins = []
        backward_operands = []

        for i, (idx, ops) in enumerate(zip(ins, forward_operands)):
            if i not in param_locs:
                backward_ins.append(idx)
                backward_operands.append(ops)
                continue

            backward_ins.append(param_symbol + idx)
            ops = cp.expand_dims(ops, 0)
            ops = cp.ascontiguousarray(cp.broadcast_to(ops, (2 * n_params, *ops.shape[1:])))
            backward_operands.append(ops)

        return ",".join(backward_ins) + "->" + out + param_symbol, backward_operands

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
        self, expr: str, operands: list[cp.ndarray], params: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Backward pass of the network.

        Args:
            expr (str): `expr` by `CircuitToEinsum`
            operands (list[cp.ndarray]): `operands` by `CircuitToEinsum`
            params (Sequence[float]): phase params for ansatz portion

        Returns:
            gradient values
        """

        expr, operands = self._prepare_backward_circuit(expr, operands)

        pname2theta = {f"{self.pname_symbol}[{i}]": params[i] for i in range(len(params))}
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
        self, expr: str, operands: list[cp.ndarray], params: Sequence[float] | np.ndarray
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
        pname2locs: dict[str, tuple[int, int, Pauli]],
        phase_shift: float = np.pi / 2,
    ):
        backups = {}
        try:
            theta = pname2theta[pname]  # e.g. pname = "θ[0]"
            loc, dag_loc, make_paulis = pname2locs[pname]
            backups = {loc: operands[loc], dag_loc: operands[dag_loc]}
            operands[loc], operands[dag_loc] = make_paulis(theta + phase_shift)
            yield operands
        finally:
            for i, v in backups.items():
                operands[i] = v
