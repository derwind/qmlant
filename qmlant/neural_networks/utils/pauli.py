from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import cupy as cp
import numpy as np

Pauli = Callable


def MatZero(xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    return xp.zeros((2, 2), dtype=dtype)


def Identity(xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    return xp.eye(2, dtype=dtype)


def PauliX(xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    return xp.array(
        [[0, 1], [1, 0]],
        dtype=dtype,
    )


def PauliY(xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    return xp.array(
        [[0, -1j], [1j, 0]],
        dtype=dtype,
    )


def PauliZ(xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    return xp.array(
        [[1, 0], [0, -1]],
        dtype=dtype,
    )


def Rx(theta: float, xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos, -sin * 1.0j], [-sin * 1.0j, cos]],
        dtype=dtype,
    )


def Ry(theta: float, xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos, -sin], [sin, cos]],
        dtype=dtype,
    )


def Rz(theta: float, xp=cp, dtype=complex) -> np.ndarray | cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
        dtype=dtype,
    )


@overload
def Rx_Rxdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Rx_Rxdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Rx_Rxdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rx_rxdag = xp.array(
            [[[cos, -sin * 1.0j], [-sin * 1.0j, cos]], [[cos, sin * 1.0j], [sin * 1.0j, cos]]],
            dtype=dtype,
        )
        return rx_rxdag[0], rx_rxdag[1]

    mat[0][0] = mat[1][1] = mat_dag[0][0] = mat_dag[1][1] = cos
    mat[0][1] = mat[1][0] = -sin * 1.0j
    mat_dag[0][1] = mat_dag[1][0] = sin * 1.0j
    return None


@overload
def Ry_Rydag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Ry_Rydag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Ry_Rydag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        ry_rydag = xp.array(
            [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
            dtype=dtype,
        )
        return ry_rydag[0], ry_rydag[1]

    mat[0][0] = mat[1][1] = mat_dag[0][0] = mat_dag[1][1] = cos
    mat[0][1] = mat_dag[1][0] = -sin
    mat[1][0] = mat_dag[0][1] = sin
    return None


@overload
def Rz_Rzdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Rz_Rzdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Rz_Rzdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rz_rzdag = xp.array(
            [
                [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
                [[cos + sin * 1.0j, 0], [0, cos - sin * 1.0j]],
            ],
            dtype=dtype,
        )
        return rz_rzdag[0], rz_rzdag[1]

    mat[0][0] = mat_dag[1][1] = cos - sin * 1.0j
    mat_dag[0][0] = mat[1][1] = cos + sin * 1.0j
    mat[0][1] = mat[1][0] = mat_dag[0][1] = mat_dag[1][0] = 0
    return None


@overload
def Rxx_Rxxdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Rxx_Rxxdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Rxx_Rxxdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rxx_rxxdag = xp.array([
            [
                [
                    [[cos, 0], [0, -sin*1j]],
                    [[0, cos], [-sin*1j, 0]]
                ],
                [
                    [[0, -sin*1j], [cos, 0]],
                    [[-sin*1j, 0], [0, cos]]
                ],
            ],
            [
                [
                    [[cos, 0], [0, sin*1j]],
                    [[0, cos], [sin*1j, 0]]
                ],
                [
                    [[0, sin*1j], [cos, 0]],
                    [[sin*1j, 0], [0, cos]]
                ],
            ],
        ], dtype=dtype)
        return rxx_rxxdag[0], rxx_rxxdag[1]

    mat[0][0][0][0] = mat[0][0][1][1] = mat[0][1][1][0] = mat[0][1][1][1] = mat_dag[0][0][0][0] = mat_dag[0][0][1][1] = mat_dag[0][1][1][0] = mat_dag[0][1][1][1] = cos
    mat[0][0][1][1] = mat[0][1][1][0] = mat[1][0][0][1] = mat[1][1][0][0] = -sin * 1.0j
    mat_dag[0][0][1][1] = mat_dag[0][1][1][0] = mat_dag[1][0][0][1] = mat_dag[1][1][0][0] = sin * 1.0j
    mat[0][0][0][1] = mat[0][0][1][0] = mat[0][1][0][0] = mat[0][1][1][1] = \
    mat[1][0][0][0] = mat[1][0][1][1] = mat[1][1][0][1] = mat[1][1][1][0] = \
    mat_dag[0][0][0][1] = mat_dag[0][0][1][0] = mat_dag[0][1][0][0] = mat_dag[0][1][1][1] = \
    mat_dag[1][0][0][0] = mat_dag[1][0][1][1] = mat_dag[1][1][0][1] = mat_dag[1][1][1][0] = 0
    return None


@overload
def Ryy_Ryydag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Ryy_Ryydag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Ryy_Ryydag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        ryy_ryydag = xp.array([
            [
                [
                    [[cos, 0], [0, sin*1j]],
                    [[0, cos], [-sin*1j, 0]]
                ],
                [
                    [[0, -sin*1j], [cos, 0]],
                    [[sin*1j, 0], [0, cos]]
                ],
            ],
            [
                [
                    [[cos, 0], [0, -sin*1j]],
                    [[0, cos], [sin*1j, 0]]
                ],
                [
                    [[0, sin*1j], [cos, 0]],
                    [[-sin*1j, 0], [0, cos]]
                ],
            ],
        ], dtype=dtype)
        return ryy_ryydag[0], ryy_ryydag[1]

    mat[0][0][0][0] = mat[0][0][1][1] = mat[0][1][1][0] = mat[0][1][1][1] = mat_dag[0][0][0][0] = mat_dag[0][0][1][1] = mat_dag[0][1][1][0] = mat_dag[0][1][1][1] = cos
    mat[0][1][1][0] = mat[1][0][0][1] = mat_dag[0][0][1][1] = mat_dag[1][1][0][0] = -sin * 1.0j
    mat[0][0][1][1] = mat[1][1][0][0] = mat_dag[0][1][1][0] = mat_dag[1][0][0][1] =  sin * 1.0j
    mat[0][0][0][1] = mat[0][0][1][0] = mat[0][1][0][0] = mat[0][1][1][1] = \
    mat[1][0][0][0] = mat[1][0][1][1] = mat[1][1][0][1] = mat[1][1][1][0] = \
    mat_dag[0][0][0][1] = mat_dag[0][0][1][0] = mat_dag[0][1][0][0] = mat_dag[0][1][1][1] = \
    mat_dag[1][0][0][0] = mat_dag[1][0][1][1] = mat_dag[1][1][0][1] = mat_dag[1][1][1][0] = 0
    return None


@overload
def Rzz_Rzzdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=..., dtype=...
) -> tuple[np.ndarray, np.ndarray] | tuple[cp.ndarray, cp.ndarray]:
    ...


@overload
def Rzz_Rzzdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=..., dtype=...) -> None:
    ...


def Rzz_Rzzdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp, dtype=complex
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rzz_rzzdag = xp.array([
            [
                [
                    [[cos - sin * 1j, 0], [0, 0]],
                    [[0, cos + sin * 1j], [0, 0]]
                ],
                [
                    [[0, 0], [cos + sin * 1j, 0]],
                    [[0, 0], [0, cos - sin * 1j]]
                ],
            ],
            [
                [
                    [[cos + sin * 1j, 0], [0, 0]],
                    [[0, cos - sin * 1j], [0, 0]]
                ],
                [
                    [[0, 0], [cos - sin * 1j, 0]],
                    [[0, 0], [0, cos + sin * 1j]]
                ],
            ],
        ], dtype=dtype)
        return rzz_rzzdag[0], rzz_rzzdag[1]

    mat[0][0][0][0] = mat[1][1][1][1] = mat_dag[0][1][0][1] = mat_dag[1][0][1][0] = cos - sin * 1.0j
    mat[0][1][0][1] = mat[1][0][1][0] = mat_dag[0][0][0][0] = mat_dag[1][1][1][1] = cos + sin * 1.0j
    mat[0][0][0][1] = mat[0][0][1][0] = mat[0][0][1][1] = mat[0][1][0][0] = mat[0][1][1][0] = \
    mat[0][1][1][1] = mat[1][0][0][0] = mat[1][0][0][1] = mat[1][0][1][1] = mat[1][1][0][0] = \
    mat[1][1][0][1] = mat[1][1][1][0] = mat_dag[0][0][0][1] = mat_dag[0][0][1][0] = mat_dag[0][0][1][1] = \
    mat_dag[0][1][0][0] = mat_dag[0][1][1][0] = mat_dag[0][1][1][1] = mat_dag[1][0][0][0] = mat_dag[1][0][0][1] = \
    mat_dag[1][0][1][1] = mat_dag[1][1][0][0] = mat_dag[1][1][0][1] = mat_dag[1][1][1][0] = 0
    return None


def PauliMatrices(
    theta: float, xp=cp, dtype=complex
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | tuple[
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
        cp.ndarray,
    ]
):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)

    rx_rxdag = xp.array(
        [[[cos, -sin * 1.0j], [-sin * 1.0j, cos]], [[cos, sin * 1.0j], [sin * 1.0j, cos]]],
        dtype=dtype,
    )

    ry_rydag = xp.array(
        [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
        dtype=dtype,
    )

    rz_rzdag = xp.array(
        [
            [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
            [[cos + sin * 1.0j, 0], [0, cos - sin * 1.0j]],
        ],
        dtype=dtype,
    )

    rxx_rxxdag = xp.array(
        [
            [
                [
                    [[cos, 0], [0, sin*1j]],
                    [[0, cos], [-sin*1j, 0]]
                ],
                [
                    [[0, -sin*1j], [cos, 0]],
                    [[sin*1j, 0], [0, cos]]
                ],
            ],
            [
                [
                    [[cos, 0], [0, -sin*1j]],
                    [[0, cos], [sin*1j, 0]]
                ],
                [
                    [[0, sin*1j], [cos, 0]],
                    [[-sin*1j, 0], [0, cos]]
                ],
            ],
        ],
        dtype=dtype,
    )

    ryy_ryydag = xp.array(
        [
            [
                [
                    [[cos, 0], [0, sin*1j]],
                    [[0, cos], [-sin*1j, 0]]
                ],
                [
                    [[0, -sin*1j], [cos, 0]],
                    [[sin*1j, 0], [0, cos]]
                ],
            ],
            [
                [
                    [[cos, 0], [0, -sin*1j]],
                    [[0, cos], [sin*1j, 0]]
                ],
                [
                    [[0, sin*1j], [cos, 0]],
                    [[-sin*1j, 0], [0, cos]]
                ],
            ],
        ],
        dtype=dtype,
    )

    rzz_rzzdag = xp.array(
        [
            [
                [
                    [[cos - sin * 1j, 0], [0, 0]],
                    [[0, cos + sin * 1j], [0, 0]]
                ],
                [
                    [[0, 0], [cos + sin * 1j, 0]],
                    [[0, 0], [0, cos - sin * 1j]]
                ],
            ],
            [
                [
                    [[cos + sin * 1j, 0], [0, 0]],
                    [[0, cos - sin * 1j], [0, 0]]
                ],
                [
                    [[0, 0], [cos - sin * 1j, 0]],
                    [[0, 0], [0, cos + sin * 1j]]
                ],
            ],
        ],
        dtype=dtype,
    )
    return (
        rx_rxdag[0],
        rx_rxdag[1],
        ry_rydag[0],
        ry_rydag[1],
        rz_rzdag[0],
        rz_rzdag[1],
        rxx_rxxdag[0],
        rxx_rxxdag[1],
        ryy_ryydag[0],
        ryy_ryydag[1],
        rzz_rzzdag[0],
        rzz_rzzdag[1],
    )
