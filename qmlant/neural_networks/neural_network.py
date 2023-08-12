from __future__ import annotations

import cupy as cp


def Ry(theta):
    return cp.array(
        [[cp.cos(theta / 2), -cp.sin(theta / 2)], [cp.sin(theta / 2), cp.cos(theta / 2)]],
        dtype=complex,
    )
