from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils import data as torch_data


class BaseDataset(torch_data.Dataset):
    def __init__(
        self,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Iris(BaseDataset):
    classes = [
        "0 - setosa",
        "1 - versicolor",
        "2 - virginica",
    ]

    def __init__(
        self,
        train: bool = True,
        test_size: float = 0.0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        subclass_targets: Sequence[int] | None = None,
    ) -> None:
        super().__init__(transform=transform, target_transform=target_transform)
        self.train = train
        self.test_size = test_size
        self.subclass_targets = subclass_targets
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        iris = datasets.load_iris()
        if isinstance(iris, tuple):
            iris = iris[0]

        if self.subclass_targets:
            indices = np.zeros_like(iris.target, dtype=bool)
            for t in self.subclass_targets:
                indices |= iris.target == t
            data = iris.data[indices]
            target = iris.target[indices]
        else:
            data = iris.data
            target = iris.target

        if 0.0 < self.test_size < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=self.test_size, random_state=1234
            )
        elif self.test_size == 0.0:
            X_train, y_train = data, target
            X_test, y_test = None, None
        elif self.test_size == 1.0:
            X_train, y_train = None, None
            X_test, y_test = data, target
        else:
            raise ValueError("test_size should be in [0.0, 1.0]")

        if self.train:
            return X_train, y_train

        return X_test, y_test

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}
