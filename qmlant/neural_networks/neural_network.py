from __future__ import annotations

import cupy as cp
import numpy as np


def Ry(theta: float, xp=cp):
    cos = xp.cos(theta / 2)
    sin = xp.sin(theta / 2)
    return xp.array(
        [[cos, -sin], [sin, cos]],
        dtype=complex,
    )


def Ry_Rydag(theta: float, xp=cp):
    cos = xp.cos(theta / 2)
    sin = xp.sin(theta / 2)
    ry_rydag = xp.array(
        [[[cos, -sin], [sin, cos]], [[cos, sin], [-sin, cos]]],
        dtype=complex,
    )
    return ry_rydag[0], ry_rydag[1]
