"""Domain adaptation interventions: augmentation, self-training, active selection."""

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data_acquisition import NUM_CLASSES, NODATA_LABEL, CLASS_NAMES
from src.dataset import LandCoverDataset
from src.training import SegmentationMetrics, evaluate, get_device

# ─── Augmentation ranges calibrated to Wasserstein distances (Phase 3) ──

# Band-specific augmentation ranges based on W-distances:
# B2: W=0.060, B3: W=0.102, B4: W=0.213, B8: W=0.035
AUG_OFFSET_RANGES = [0.06, 0.10, 0.15, 0.04]  # per-band max offset
AUG_SCALE_RANGES = [(0.7, 1.3), (0.6, 1.4), (0.5, 1.5), (0.8, 1.2)]  # per-band


# ─── 4a: Photometric augmentation dataset ────────────────────────────

def _recompute_ndvi(image):
    """Recompute NDVI (band 4) from B8 (band 3) and B4 (band 2)."""
    b8, b4 = image[3], image[2]
    denom = b8 + b4
    image[4] = np.where(denom > 1e-8, (b8 - b4) / denom, 0.0)
    image[4] = np.clip(image[4], 0, 1)
    return image


def _photometric_transform(image, target_chip_quantiles=None, src_quantiles=None):
    """Photometric augmentations on raw (unnormalized) image (5, H, W).

    Always recomputes NDVI after modifying reflectance bands.
    """
    # Per-band brightness offset (bands 0-3)
    if random.random() < 0.5:
        for b in range(4):
            offset = random.uniform(-AUG_OFFSET_RANGES[b], AUG_OFFSET_RANGES[b])
            image[b] = image[b] + offset

    # Per-band contrast scale (bands 0-3)
    if random.random() < 0.5:
        for b in range(4):
            lo, hi = AUG_SCALE_RANGES[b]
            scale = random.uniform(lo, hi)
            image[b] = image[b] * scale

    # Gamma correction (bands 0-3)
    if random.random() < 0.3:
        for b in range(4):
            gamma = random.uniform(0.7, 1.3)
            image[b] = np.clip(image[b], 0, None) ** gamma

    # Recompute NDVI after band transforms
    image[:4] = np.clip(image[:4], 0, 1)
    image = _recompute_ndvi(image)

    # Gaussian blur (50%)
    if random.random() < 0.5:
        sigma = random.uniform(0.5, 1.5)
        for b in range(5):
            image[b] = gaussian_filter(image[b], sigma=sigma)

    # Histogram match to random target chip (30%)
    if target_chip_quantiles is not None and src_quantiles is not None:
        if random.random() < 0.3:
            chip_idx = random.randint(0, len(target_chip_quantiles) - 1)
            tgt_q = target_chip_quantiles[chip_idx]  # (4, n_q)
            for b in range(4):
                band = image[b].ravel()
                # Rank in source distribution → map to target quantiles
                ranks = np.interp(band, src_quantiles[b],
                                  np.linspace(0, 1, len(src_quantiles[b])))
                matched = np.interp(ranks,
                                    np.linspace(0, 1, len(tgt_q[b])), tgt_q[b])
                image[b] = matched.reshape(image[b].shape)
            image[:4] = np.clip(image[:4], 0, 1)
            image = _recompute_ndvi(image)

    return image


def _geo_augment(image, label):
    """Random h-flip, v-flip, 90-degree rotations."""
    if random.random() > 0.5:
        image = image[:, :, ::-1]
        label = label[:, ::-1]
    if random.random() > 0.5:
        image = image[:, ::-1, :]
        label = label[::-1, :]
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(0, 1))
    return image, label


def precompute_chip_quantiles(domain_dir, chip_indices, n_quantiles=100):
    """Precompute per-chip quantiles for bands 0-3 (reflectance bands only)."""
    domain_dir = Path(domain_dir)
    all_quantiles = []
    for idx in chip_indices:
        img = np.load(domain_dir / "images" / f"chip_{idx:03d}.npy")
        chip_q = np.zeros((4, n_quantiles))
        qs = np.linspace(0, 1, n_quantiles)
        for b in range(4):
            band = img[b].ravel()
            valid = (band > -0.5) & np.isfinite(band)
            if valid.sum() > 10:
                chip_q[b] = np.quantile(band[valid], qs)
        all_quantiles.append(chip_q)
    return all_quantiles  # list of (4, n_quantiles) arrays


class PhotoAugDataset(Dataset):
    """Source dataset with photometric + geometric augmentations."""

    def __init__(self, domain_dir, chip_indices, mean, std,
                 target_chip_quantiles=None, src_quantiles=None):
        self.domain_dir = Path(domain_dir)
        self.chip_indices = chip_indices
        self.mean = np.array(mean, dtype=np.float32).reshape(5, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(5, 1, 1)
        self.target_chip_quantiles = target_chip_quantiles
        self.src_quantiles = src_quantiles

    def __len__(self):
        return len(self.chip_indices)

    def __getitem__(self, idx):
        chip_idx = self.chip_indices[idx]
        image = np.load(self.domain_dir / "images" / f"chip_{chip_idx:03d}.npy").copy()
        label = np.load(self.domain_dir / "labels" / f"chip_{chip_idx:03d}.npy").copy()

        # Geo augmentations first
        image, label = _geo_augment(image, label)

        # Photometric on raw values
        image = _photometric_transform(image, self.target_chip_quantiles, self.src_quantiles)

        # Normalize
        image = (image - self.mean) / (self.std + 1e-8)
        image = np.nan_to_num(image, nan=0.0)

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


# ─── 4b: Self-training with mean teacher ─────────────────────────────

def update_ema(student, teacher, alpha=0.99):
    """EMA update of teacher from student, including BN buffers."""
    student_sd = student.state_dict()
    teacher_sd = teacher.state_dict()
    for key in teacher_sd:
        teacher_sd[key] = alpha * teacher_sd[key] + (1 - alpha) * student_sd[key]
    teacher.load_state_dict(teacher_sd)


@torch.no_grad()
def compute_per_class_thresholds(teacher, target_loader, device, quantile=0.7):
    """Per-class confidence thresholds from teacher predictions."""
    teacher.eval()
    per_class_confs = [[] for _ in range(NUM_CLASSES)]

    for batch in target_loader:
        images = batch[0].to(device)
        probs = F.softmax(teacher(images), dim=1)
        max_probs, preds = probs.max(dim=1)

        max_probs = max_probs.cpu().numpy().ravel()
        preds = preds.cpu().numpy().ravel()

        for c in range(NUM_CLASSES):
            mask = preds == c
            if mask.any():
                per_class_confs[c].append(max_probs[mask])

    thresholds = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if per_class_confs[c]:
            all_confs = np.concatenate(per_class_confs[c])
            if len(all_confs) >= 100:
                thresholds[c] = float(np.quantile(all_confs, quantile))
            else:
                thresholds[c] = 0.9
        else:
            thresholds[c] = 1.0  # class never predicted → block
    return thresholds


@torch.no_grad()
def _compute_pseudo_label_accuracy(pseudo_labels, true_labels):
    """Compare pseudo-labels to true labels (logging only, never used for training)."""
    valid = (true_labels != NODATA_LABEL) & (pseudo_labels != NODATA_LABEL)
    if valid.sum() == 0:
        return 0.0, 0
    correct = (pseudo_labels[valid] == true_labels[valid]).sum()
    return float(correct) / float(valid.sum()), int(valid.sum())


@dataclass
class SelfTrainConfig:
    epochs: int = 15
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 1e-4
    ema_alpha: float = 0.99
    confidence_quantile: float = 0.7
    lambda_u_max: float = 0.5
    rampup_epochs: int = 3
    device: str = "mps"
    num_workers: int = 8
    save_dir: str = "checkpoints/4b"


def run_self_training(
    student, source_loader, target_loader, val_loader, eval_loader,
    config, class_weights=None,
):
    """Full mean-teacher self-training loop.

    Teacher sees clean target images; student sees photometric-augmented version.
    Saves last-epoch teacher (fixed schedule, not best-target-eval).
    """
    device = get_device(config.device)
    student = student.to(device)
    teacher = copy.deepcopy(student)
    teacher.eval()
    teacher = teacher.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        ignore_index=NODATA_LABEL,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_losses": [], "src_mious": [], "tgt_mious": [],
        "pseudo_accuracies": [], "pseudo_coverages": [],
        "target_class_histograms": [],
    }

    for epoch in range(config.epochs):
        # Lambda ramp-up
        lam = min(1.0, (epoch + 1) / config.rampup_epochs) * config.lambda_u_max

        # Compute per-class thresholds
        thresholds = compute_per_class_thresholds(
            teacher, target_loader, device, config.confidence_quantile
        ).to(device)

        # Training
        student.train()
        total_loss = 0.0
        n_batches = 0
        pseudo_correct_total = 0
        pseudo_count_total = 0
        target_pred_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        src_iter = iter(source_loader)
        pbar = tqdm(target_loader, desc=f"Self-train {epoch+1}/{config.epochs}", leave=False)

        for tgt_batch in pbar:
            # Get source batch (cycle if exhausted)
            try:
                src_images, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                src_images, src_labels = next(src_iter)

            src_images = src_images.to(device)
            src_labels = src_labels.to(device)
            tgt_images = tgt_batch[0].to(device)
            tgt_true_labels = tgt_batch[1].numpy()  # for logging only

            # Teacher generates pseudo-labels on clean target images
            with torch.no_grad():
                teacher.eval()
                teacher_probs = F.softmax(teacher(tgt_images), dim=1)
                teacher_conf, teacher_preds = teacher_probs.max(dim=1)

                # Per-class threshold masking
                pseudo_labels = teacher_preds.clone()
                for c in range(NUM_CLASSES):
                    low_conf = (teacher_preds == c) & (teacher_conf < thresholds[c])
                    pseudo_labels[low_conf] = NODATA_LABEL

            # Concat source + target for stable BN, forward student
            student.train()
            combined = torch.cat([src_images, tgt_images], dim=0)
            combined_logits = student(combined)
            src_logits = combined_logits[:len(src_images)]
            tgt_logits = combined_logits[len(src_images):]

            src_loss = criterion(src_logits, src_labels)
            pseudo_loss = criterion(tgt_logits, pseudo_labels)
            loss = src_loss + lam * pseudo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update (state_dict includes BN buffers)
            update_ema(student, teacher, config.ema_alpha)

            total_loss += loss.item()
            n_batches += 1

            # Log pseudo-label accuracy (compare to true labels on disk)
            pl_np = pseudo_labels.cpu().numpy()
            acc, cnt = _compute_pseudo_label_accuracy(pl_np, tgt_true_labels)
            pseudo_correct_total += int(acc * cnt)
            pseudo_count_total += cnt

            # Log target class predictions
            tgt_preds_np = teacher_preds.cpu().numpy().ravel()
            for c in range(NUM_CLASSES):
                target_pred_counts[c] += (tgt_preds_np == c).sum()

            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", lam=f"{lam:.2f}")

        # Epoch stats
        epoch_loss = total_loss / max(n_batches, 1)
        pseudo_acc = pseudo_correct_total / max(pseudo_count_total, 1)
        pseudo_coverage = pseudo_count_total / max(
            target_pred_counts.sum(), 1
        )  # fraction of target pixels that passed thresholds

        # Target class histogram (collapse detection)
        tgt_class_hist = target_pred_counts / max(target_pred_counts.sum(), 1)
        max_class_frac = tgt_class_hist.max()

        # Evaluate teacher on both domains
        teacher.eval()
        _, _, src_miou, _, _ = evaluate(teacher, val_loader, criterion, device)
        _, _, tgt_miou, _, _ = evaluate(teacher, eval_loader, criterion, device)

        history["train_losses"].append(epoch_loss)
        history["src_mious"].append(src_miou)
        history["tgt_mious"].append(tgt_miou)
        history["pseudo_accuracies"].append(pseudo_acc)
        history["pseudo_coverages"].append(pseudo_coverage)
        history["target_class_histograms"].append(tgt_class_hist.tolist())

        collapse_warn = " *** COLLAPSE WARNING ***" if max_class_frac > 0.9 else ""
        print(
            f"  Epoch {epoch+1}/{config.epochs} | loss={epoch_loss:.4f} | "
            f"src_mIoU={src_miou:.4f} | tgt_mIoU={tgt_miou:.4f} | "
            f"pseudo_acc={pseudo_acc:.3f} | coverage={pseudo_coverage:.3f} | "
            f"lambda={lam:.2f}{collapse_warn}"
        )

    # Save last-epoch teacher (fixed schedule)
    torch.save(teacher.state_dict(), save_dir / "best_model.pth")
    print(f"  Saved last-epoch teacher to {save_dir / 'best_model.pth'}")

    return history, teacher


# ─── 4c: Active selection ────────────────────────────────────────────

@torch.no_grad()
def compute_chip_entropy(model, domain_dir, chip_indices, mean, std, device):
    """Mean predictive entropy per chip."""
    ds = LandCoverDataset(domain_dir, chip_indices, mean, std, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8)

    entropies = {}
    chip_ptr = 0
    for images, labels in tqdm(loader, desc="Computing entropy", leave=False):
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        H = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        for i in range(images.shape[0]):
            valid = labels[i].numpy() != NODATA_LABEL
            if valid.any():
                entropies[chip_indices[chip_ptr]] = float(H[i].cpu().numpy()[valid].mean())
            else:
                entropies[chip_indices[chip_ptr]] = 0.0
            chip_ptr += 1

    return entropies


def select_active_chips(model, domain_dir, candidates, mean, std, device,
                        n_select=30, pool_size=90, seed=42):
    """Select chips by entropy + k-means diversity."""
    from src.diagnostics import extract_embeddings

    embeddings = extract_embeddings(model, domain_dir, candidates, mean, std, device)
    entropies = compute_chip_entropy(model, domain_dir, candidates, mean, std, device)

    idx_to_pos = {idx: i for i, idx in enumerate(candidates)}

    # Top pool_size by entropy
    sorted_chips = sorted(candidates, key=lambda x: entropies.get(x, 0), reverse=True)
    pool = sorted_chips[:pool_size]

    # K-means on pool embeddings
    pool_emb = np.stack([embeddings[idx_to_pos[idx]] for idx in pool])
    kmeans = KMeans(n_clusters=n_select, random_state=seed, n_init=10)
    kmeans.fit(pool_emb)

    selected = []
    for c in range(n_select):
        cluster_mask = kmeans.labels_ == c
        cluster_chips = [pool[i] for i in range(len(pool)) if cluster_mask[i]]
        cluster_entropies = [entropies[idx] for idx in cluster_chips]
        best = cluster_chips[np.argmax(cluster_entropies)]
        selected.append(best)

    return selected


def select_random_chips(candidates, n_select=30, seed=42):
    """Random baseline."""
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, n_select, replace=False).tolist()


# ─── 4c: Mixed domain dataset ───────────────────────────────────────

class MixedDomainDataset(Dataset):
    """Source + labeled target chips with oversampling."""

    def __init__(self, source_dir, source_indices, target_dir, target_indices,
                 mean, std, target_oversample=8, augment=True):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.mean = np.array(mean, dtype=np.float32).reshape(5, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(5, 1, 1)
        self.augment = augment

        self.items = [(self.source_dir, idx) for idx in source_indices]
        for _ in range(target_oversample):
            self.items.extend([(self.target_dir, idx) for idx in target_indices])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        domain_dir, chip_idx = self.items[idx]
        image = np.load(domain_dir / "images" / f"chip_{chip_idx:03d}.npy")
        label = np.load(domain_dir / "labels" / f"chip_{chip_idx:03d}.npy")

        image = (image - self.mean) / (self.std + 1e-8)
        image = np.nan_to_num(image, nan=0.0)

        if self.augment:
            image, label = _geo_augment(image, label)

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()
