from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from crgan.data import FewShotDataset
from crgan.models import Discriminator, Generator


class CRGANSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            latent_dim=100,
            img_size=64,
            channels=1,
            classes=6,
        )

    def test_model_shapes(self) -> None:
        generator = Generator(self.args).eval()
        discriminator = Discriminator(self.args).eval()
        with torch.inference_mode():
            images = generator(torch.randn(2, self.args.latent_dim))
            validity, features, style = discriminator(images)
        self.assertEqual(tuple(images.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(validity.shape), (2, 1))
        self.assertEqual(tuple(features[0].shape), (2, 100))
        self.assertEqual(tuple(features[1].shape), (2, 100))
        self.assertEqual(tuple(style.shape), (2, 512))

    def test_balanced_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name in ("a", "b"):
                class_dir = root / class_name
                class_dir.mkdir()
                for index in range(3):
                    Image.new("RGB", (8, 8), color=(index, index, index)).save(
                        class_dir / f"{index}.png"
                    )
            dataset = FewShotDataset(root, num_samples=4, seed=7)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.count.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
