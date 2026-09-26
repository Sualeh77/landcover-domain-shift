"""PyTorch Dataset for loading and augmenting land-cover chips."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    def __init__(
        self,
        domain_dir: str | Path,
        chip_indices: list[int],
        mean: list[float],
        std: list[float],
        augment: bool = False,
    ):
        self.domain_dir = Path(domain_dir)
        self.chip_indices = chip_indices
        self.mean = np.array(mean, dtype=np.float32).reshape(5, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(5, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.chip_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chip_idx = self.chip_indices[idx]
        image = np.load(self.domain_dir / "images" / f"chip_{chip_idx:03d}.npy")
        label = np.load(self.domain_dir / "labels" / f"chip_{chip_idx:03d}.npy")

        image = (image - self.mean) / (self.std + 1e-8)
        image = np.nan_to_num(image, nan=0.0)

        if self.augment:
            image, label = self._augment(image, label)

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy()).long()

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray):
        # image: (C, H, W), label: (H, W)
        if random.random() > 0.5:
            image = image[:, :, ::-1]  # h-flip: flip axis=2
            label = label[:, ::-1]     # h-flip: flip axis=1

        if random.random() > 0.5:
            image = image[:, ::-1, :]  # v-flip: flip axis=1
            label = label[::-1, :]     # v-flip: flip axis=0

        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))

        return image, label


def load_norm_stats(stats_path: str | Path) -> tuple[list[float], list[float]]:
    with open(stats_path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]
