import os
from typing import Any
from urllib.error import URLError

import numpy as np
from PIL import Image
from torchvision import datasets


def download_file(
    url: str,
    download_root: str,
    filename: str | None = None,
    md5: str | None = None,
) -> None:
    download_root = os.path.expanduser(download_root)
    if not filename:
        filename = os.path.basename(url)

    datasets.utils.download_url(url, download_root, filename, md5)


class KMNIST49(datasets.mnist.MNIST):
    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/k49/"]

    resources = [
        ("k49-train-imgs.npz", "7ac088b20481cf51dcd01ceaab89d821"),
        ("k49-train-labels.npz", "44a8e1b893f81e63ff38d73cad420f7a"),
        ("k49-test-imgs.npz", "d352e201d846ce6b94f42c990966f374"),
        ("k49-test-labels.npz", "4da6f7a62e67a832d5eb1bd85c5ee448"),
    ]
    classes = [
        "a",
        "i",
        "u",
        "e",
        "o",
        "ka",
        "ki",
        "ku",
        "ke",
        "ko",
        "sa",
        "si",
        "su",
        "se",
        "so",
        "ta",
        "ti",
        "tu",
        "te",
        "to",
        "na",
        "ni",
        "nu",
        "ne",
        "no",
        "ha",
        "hi",
        "hu",
        "he",
        "ho",
        "ma",
        "mi",
        "mu",
        "me",
        "mo",
        "ya",
        "yu",
        "yo",
        "ra",
        "ri",
        "ru",
        "re",
        "ro",
        "wa",
        "wi",
        "we",
        "wo",
        "nn",
        "dou",
    ]

    def _load_data(self):
        data_file = f"k49-{'train' if self.train else 'test'}-imgs.npz"
        data = np.load(os.path.join(self.raw_folder, data_file))["arr_0"]

        label_file = f"k49-{'train' if self.train else 'test'}-labels.npz"
        targets = np.load(os.path.join(self.raw_folder, label_file))["arr_0"]

        return data, targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self) -> bool:
        return all(
            datasets.utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url)))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_file(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")
