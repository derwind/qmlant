from __future__ import annotations

from typing import Literal, overload

import cupy as cp
import numpy as np


def Ry(theta: float, xp=cp) -> cp.ndarray:
    cos = xp.cos(theta / 2)
    sin = xp.sin(theta / 2)
    return xp.array(
        [[cos, -sin], [sin, cos]],
        dtype=complex,
    )


@overload
def Ry_Rydag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=...
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
    ...


@overload
def Ry_Rydag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=...) -> None:
    ...


def Ry_Rydag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = xp.cos(theta / 2)
    sin = xp.sin(theta / 2)
    if mat is None or mat_dag is None:
        ry_rydag = xp.array(
            [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
            dtype=complex,
        )
        return ry_rydag[0], ry_rydag[1]

    mat[0][0] = mat[1][1] = mat_dag[0][0] = mat_dag[1][1] = cos
    mat[0][1] = mat_dag[1][0] = -sin
    mat[1][0] = mat_dag[0][1] = sin
    return None
