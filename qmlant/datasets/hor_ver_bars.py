from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import train_test_split


class HorVerBars:
    classes = [
        "-1 - horizontal",
        "1 - vertical",
    ]

    def __init__(
        self,
        train: bool = True,
        test_size: float = 0.3,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test_size = test_size
        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        images, labels = generate_dataset(50)
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=self.test_size
        )

        if self.train:
            return train_images, train_labels

        return test_images, test_labels

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


# https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html
def generate_dataset(num_images):
    from qiskit.utils import algorithm_globals

    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for _ in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels
