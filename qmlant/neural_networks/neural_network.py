from __future__ import annotations

import cupy as cp


def Ry(theta: float):
    cos = cp.cos(theta / 2)
    sin = cp.sin(theta / 2)
    return cp.array(
        [[cos, -sin], [sin, cos]],
        dtype=complex,
    )


def Ry_Rydag(theta: float):
    cos = cp.cos(theta / 2)
    sin = cp.sin(theta / 2)
    ry_rydag = cp.array(
        [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
        dtype=complex,
    )
    return ry_rydag[0], ry_rydag[1]
