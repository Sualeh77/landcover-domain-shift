"""Training loop, eval metrics, and model creation."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from src.data_acquisition import NUM_CLASSES, CLASS_NAMES, NODATA_LABEL


# ─── Model creation ─────────────────────────────────────────────────

def create_model(
    encoder_name: str = "resnet34",
    in_channels: int = 5,
    num_classes: int = NUM_CLASSES,
    encoder_weights: str = "imagenet",
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


# ─── Class weights ───────────────────────────────────────────────────

def compute_class_weights(
    class_freq: list[float],
    max_weight: float = 10.0,
    min_freq: float = 0.001,
) -> torch.Tensor:
    freq = np.array(class_freq, dtype=np.float64)
    weights = np.ones_like(freq)

    present = freq >= min_freq
    if present.any():
        inv_freq = 1.0 / freq[present]
        inv_freq = np.minimum(inv_freq, max_weight)
        inv_freq = inv_freq / inv_freq.mean()  # mean-normalize among present classes
        weights[present] = inv_freq

    return torch.tensor(weights, dtype=torch.float32)


# ─── Metrics ─────────────────────────────────────────────────────────

@dataclass
class SegmentationMetrics:
    num_classes: int = NUM_CLASSES
    confusion_matrix: np.ndarray = field(default=None)

    def __post_init__(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update_fast(self, preds: np.ndarray, labels: np.ndarray) -> None:
        valid = labels != NODATA_LABEL
        p = preds[valid].astype(np.int64).ravel()
        l = labels[valid].astype(np.int64).ravel()
        indices = l * self.num_classes + p
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_iou(self) -> tuple[np.ndarray, float, list[int]]:
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        per_class_iou = np.full(self.num_classes, np.nan)
        present = union > 0
        per_class_iou[present] = intersection[present] / union[present]

        present_classes = list(np.where(present)[0])
        mean_iou = float(np.nanmean(per_class_iou[present])) if present.any() else 0.0
        return per_class_iou, mean_iou, present_classes

    def reset(self) -> None:
        self.confusion_matrix[:] = 0


# ─── Training config ────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    device: str = "mps"
    save_dir: str = "checkpoints"


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Training loop ──────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train {epoch+1}/{total_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, float, np.ndarray, list[int]]:
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        metrics.update_fast(preds, labels_np)

    val_loss = total_loss / max(n_batches, 1)
    per_class_iou, mean_iou, present_classes = metrics.compute_iou()
    return val_loss, per_class_iou, mean_iou, metrics.confusion_matrix, present_classes


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> dict:
    device = get_device(config.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        ignore_index=NODATA_LABEL,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_losses": [],
        "val_losses": [],
        "val_mious": [],
        "best_epoch": 0,
        "best_miou": 0.0,
    }

    for epoch in range(config.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, total_epochs=config.epochs,
        )
        val_loss, per_class_iou, mean_iou, _, present = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_mious"].append(mean_iou)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        present_names = [CLASS_NAMES[i] for i in present]
        print(
            f"Epoch {epoch+1:2d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mIoU={mean_iou:.4f} (over {len(present)} classes) | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        iou_str = " | ".join(
            f"{CLASS_NAMES[i]}={per_class_iou[i]:.3f}" if not np.isnan(per_class_iou[i])
            else f"{CLASS_NAMES[i]}=N/A"
            for i in range(NUM_CLASSES)
        )
        print(f"  Per-class: {iou_str}")

        if mean_iou > history["best_miou"]:
            history["best_miou"] = mean_iou
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ** New best (mIoU={mean_iou:.4f}) **")

    torch.save(model.state_dict(), save_dir / "final_model.pth")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history
