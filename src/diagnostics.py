"""Probe datasets and helpers for diagnosing domain shift."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data_acquisition import NODATA_LABEL, NODATA_REFLECTANCE, load_chip


# ─── Probe datasets ─────────────────────────────────────────────────

class PerChipStandardizedDataset(Dataset):
    """Z-score each chip independently per band."""

    def __init__(self, domain_dir, chip_indices):
        self.domain_dir = Path(domain_dir)
        self.chip_indices = chip_indices

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        chip_idx = self.chip_indices[idx]
        image = np.load(self.domain_dir / "images" / f"chip_{chip_idx:03d}.npy")
        label = np.load(self.domain_dir / "labels" / f"chip_{chip_idx:03d}.npy")

        chip_mean = image.mean(axis=(1, 2), keepdims=True)
        chip_std = image.std(axis=(1, 2), keepdims=True)
        image = (image - chip_mean) / (chip_std + 1e-8)
        image = np.nan_to_num(image, nan=0.0)

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


class HistogramMatchedDataset(Dataset):
    """Global per-band quantile matching from target to source distribution.

    Precompute src/tgt quantiles, then map each target pixel:
    target_value → rank in target CDF → corresponding source value.
    Recomputes NDVI from matched B4/B8 for consistency.
    """

    def __init__(self, domain_dir, chip_indices, src_quantiles, tgt_quantiles,
                 norm_mean, norm_std, n_bands_to_match=4):
        self.domain_dir = Path(domain_dir)
        self.chip_indices = chip_indices
        self.src_q_values = src_quantiles  # (5, n_quantiles) or (4, n_quantiles)
        self.tgt_q_values = tgt_quantiles  # (5, n_quantiles) or (4, n_quantiles)
        self.norm_mean = np.array(norm_mean, dtype=np.float32).reshape(5, 1, 1)
        self.norm_std = np.array(norm_std, dtype=np.float32).reshape(5, 1, 1)
        self.n_match = n_bands_to_match  # match B2,B3,B4,B8 (first 4), recompute NDVI

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        chip_idx = self.chip_indices[idx]
        image = np.load(self.domain_dir / "images" / f"chip_{chip_idx:03d}.npy")
        label = np.load(self.domain_dir / "labels" / f"chip_{chip_idx:03d}.npy")

        # Match first 4 bands (B2, B3, B4, B8) via quantile mapping
        for b in range(self.n_match):
            band = image[b].ravel()
            # Find rank of each pixel in target distribution
            ranks = np.interp(band, self.tgt_q_values[b], np.linspace(0, 1, len(self.tgt_q_values[b])))
            # Map rank to source value
            matched = np.interp(ranks, np.linspace(0, 1, len(self.src_q_values[b])), self.src_q_values[b])
            image[b] = matched.reshape(image[b].shape)

        # Recompute NDVI from matched B8 (index 3) and B4 (index 2)
        b8 = image[3]
        b4 = image[2]
        denom = b8 + b4
        image[4] = np.where(denom > 0, (b8 - b4) / denom, 0.0)
        image[4] = np.clip(image[4], 0, 1)

        # Apply standard source normalization
        image = (image - self.norm_mean) / (self.norm_std + 1e-8)
        image = np.nan_to_num(image, nan=0.0)

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


# ─── Pixel collection ───────────────────────────────────────────────

def collect_raw_pixels(domain_dir, chip_indices, max_pixels=500_000, seed=42):
    """Load raw unnormalized pixels, exclude nodata, subsample."""
    domain_dir = Path(domain_dir)
    all_pixels = []
    for idx in tqdm(chip_indices, desc=f"Loading {domain_dir.name} pixels"):
        img, _ = load_chip(domain_dir, idx)
        pixels = img.reshape(5, -1).T  # (65536, 5)
        # Exclude nodata pixels
        valid = (pixels[:, 0] > NODATA_REFLECTANCE) & np.isfinite(pixels).all(axis=1)
        all_pixels.append(pixels[valid])
    all_pixels = np.concatenate(all_pixels, axis=0)
    if len(all_pixels) > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pixels), max_pixels, replace=False)
        all_pixels = all_pixels[idx]
    return all_pixels


def compute_quantiles(pixels, n_quantiles=1000):
    """Compute per-band quantile values from pixel array (N, 5)."""
    qs = np.linspace(0, 1, n_quantiles)
    q_values = np.zeros((pixels.shape[1], n_quantiles))
    for b in range(pixels.shape[1]):
        q_values[b] = np.quantile(pixels[:, b], qs)
    return q_values


# ─── Embedding extraction ───────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, domain_dir, chip_indices, mean, std, device, batch_size=8):
    """Extract GAP bottleneck embeddings (512-dim) for all chips."""
    from src.dataset import LandCoverDataset
    ds = LandCoverDataset(domain_dir, chip_indices, mean, std, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)

    embeddings = []
    for images, _ in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = model.encoder(images)
        bottleneck = features[-1]
        pooled = F.adaptive_avg_pool2d(bottleneck, 1).squeeze(-1).squeeze(-1)
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


# ─── Prior correction (H2 probe) ────────────────────────────────────

@torch.no_grad()
def evaluate_with_prior_correction(model, loader, device, src_prior, tgt_prior):
    """Evaluate model with oracle prior correction on softmax outputs.

    Rescale softmax by (tgt_prior / src_prior) per class, renormalize, argmax.
    Returns (per_class_iou, mean_iou, present_classes).
    """
    from src.training import SegmentationMetrics

    # Compute correction weights
    correction = np.array(tgt_prior) / (np.array(src_prior) + 1e-10)
    correction_t = torch.tensor(correction, dtype=torch.float32, device=device)

    metrics = SegmentationMetrics()
    for images, labels in tqdm(loader, desc="Prior-corrected eval"):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Rescale by prior ratio
        corrected = probs * correction_t.view(1, -1, 1, 1)
        corrected = corrected / corrected.sum(dim=1, keepdim=True)

        preds = corrected.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()
        metrics.update_fast(preds, labels_np)

    per_class_iou, mean_iou, present = metrics.compute_iou()
    return per_class_iou, mean_iou, present
