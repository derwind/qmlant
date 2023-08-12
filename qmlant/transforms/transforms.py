from __future__ import annotations

import torch


class ToTensor:
    def __init__(self, dtype=float):
        self.dtype = dtype

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)  # pylint: disable=no-member

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MapLabel:
    def __init__(self, src_classes: list[int], dst_classes: list[int]):
        if len(src_classes) != len(dst_classes):
            raise ValueError("'src_classes' and 'dst_classes' must have same size")
        self.labels_map = dict(zip(src_classes, dst_classes))

    def __call__(self, x):
        return self.labels_map[x]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
