from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import cupy as cp
import numpy as np

Pauli = Callable


def Rx(theta: float, xp=cp) -> cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos, -sin * 1.0j], [-sin * 1.0j, cos]],
        dtype=complex,
    )


def Ry(theta: float, xp=cp) -> cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos, -sin], [sin, cos]],
        dtype=complex,
    )


def Rz(theta: float, xp=cp) -> cp.ndarray:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return xp.array(
        [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
        dtype=complex,
    )


@overload
def Rx_Rxdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=...
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
    ...


@overload
def Rx_Rxdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=...) -> None:
    ...


def Rx_Rxdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rx_rxdag = xp.array(
            [[[cos, -sin * 1.0j], [-sin * 1.0j, cos]], [[cos, sin * 1.0j], [sin * 1.0j, cos]]],
            dtype=complex,
        )
        return rx_rxdag[0], rx_rxdag[1]

    mat[0][0] = mat[1][1] = mat_dag[0][0] = mat_dag[1][1] = cos
    mat[0][1] = mat[1][0] = -sin * 1.0j
    mat_dag[0][1] = mat_dag[1][0] = sin * 1.0j
    return None


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
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
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


@overload
def Rz_Rzdag(  # type: ignore
    theta: float, mat: Literal[None] = ..., mat_dag: Literal[None] = ..., xp=...
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
    ...


@overload
def Rz_Rzdag(theta: float, mat: cp.ndarray = ..., mat_dag: cp.ndarray = ..., xp=...) -> None:
    ...


def Rz_Rzdag(
    theta: float, mat: cp.ndarray | None = None, mat_dag: cp.ndarray | None = None, xp=cp
) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray] | None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    if mat is None or mat_dag is None:
        rz_rzdag = xp.array(
            [
                [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
                [[cos + sin * 1.0j, 0], [0, cos - sin * 1.0j]],
            ],
            dtype=complex,
        )
        return rz_rzdag[0], rz_rzdag[1]

    mat[0][0] = mat_dag[1][1] = cos - sin * 1.0j
    mat_dag[0][0] = mat[1][1] = cos + sin * 1.0j
    mat[0][1] = mat[1][0] = mat_dag[0][1] = mat_dag[1][0] = 0
    return None


def Rx_Rxdag_Ry_Rydag_Rz_Rzdag(
    theta: float, xp=cp
) -> tuple[
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
]:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)

    rx_rxdag = xp.array(
        [[[cos, -sin * 1.0j], [-sin * 1.0j, cos]], [[cos, sin * 1.0j], [sin * 1.0j, cos]]],
        dtype=complex,
    )

    ry_rydag = xp.array(
        [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
        dtype=complex,
    )

    rz_rzdag = xp.array(
        [
            [[cos - sin * 1.0j, 0], [0, cos + sin * 1.0j]],
            [[cos + sin * 1.0j, 0], [0, cos - sin * 1.0j]],
        ],
        dtype=complex,
    )
    return rx_rxdag[0], rx_rxdag[1], ry_rydag[0], ry_rydag[1], rz_rzdag[0], rz_rzdag[1]
