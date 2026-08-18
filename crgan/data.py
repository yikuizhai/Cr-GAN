"""ImageFolder-compatible balanced few-shot dataset loading."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import DatasetFolder


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def make_fewshot_dataset(
    directory: str | os.PathLike[str],
    class_to_idx: dict[str, int],
    samples_per_class: int,
    seed: int = 0,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Select at most ``samples_per_class`` images from every class."""
    rng = random.Random(seed)
    instances: list[tuple[str, int]] = []
    counts = np.zeros(len(class_to_idx), dtype=np.int64)

    for class_name in sorted(class_to_idx):
        class_index = class_to_idx[class_name]
        class_dir = Path(directory) / class_name
        if not class_dir.is_dir():
            continue
        files = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        rng.shuffle(files)
        for path in files[:samples_per_class]:
            instances.append((str(path), class_index))
            counts[class_index] += 1

    return instances, counts


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as handle:
        return Image.open(handle).convert("RGB")


class FewShotDataset(DatasetFolder):
    """Balanced subset of an ImageFolder-style dataset.

    ``num_samples`` is the requested total sample count. It is divided evenly
    across the discovered classes, matching the behavior of the research code.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        num_samples: int | None = None,
        transform=None,
        target_transform=None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            root=str(root),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )
        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError("num_samples must be positive")
            samples_per_class = max(1, num_samples // len(self.classes))
            self.samples, self.count = make_fewshot_dataset(
                root,
                self.class_to_idx,
                samples_per_class,
                seed=seed,
            )
            self.imgs = self.samples
            self.targets = [target for _, target in self.samples]


# Backward-compatible name used by the original training script.
ltdataset = FewShotDataset
