import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset.dataset import BraTSDataset
from models.hybrid_model import HybridUNetTransformer

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =====================================
# Config — all knobs in one place
# =====================================

CONFIG = {
    "data_path": "data",
    "image_size": (96, 96, 96),
    "num_classes": 5,
    "in_channels": 4,
    "batch_size": 1,
    "num_workers": 2,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.2,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 10,
    "use_amp": True,           # mixed precision — big speedup on GPU, ignored on CPU
    "seed": 42,
    "checkpoint_path": "best_model.pth",
    "last_checkpoint_path": "last_checkpoint.pth",
    "resume": False,           # set True to continue from last_checkpoint_path
}

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Edema",
    3: "Enhancing Tumor",
    4: "Other",
}


def set_seed(seed):
    """Reproducible runs — same split, same init, same shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================
# Dice Loss (unchanged math, cleaned up)
# =====================================

def dice_loss(prediction, target, num_classes=5, eps=1e-5):
    prediction = torch.softmax(prediction, dim=1)

    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

    intersection = (prediction * target_one_hot).sum(dim=(2, 3, 4))
    union = prediction.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class.mean(), dice_per_class


@torch.no_grad()
def per_class_dice(prediction, target, num_classes=5):
    """Dice score (not loss) per class, for readable validation metrics.
    Returns a tensor of shape (num_classes,) averaged over the batch.
    """
    pred_labels = torch.argmax(prediction, dim=1)
    dices = []
    for c in range(num_classes):
        pred_c = (pred_labels == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        dices.append(dice.item())
    return dices


# =====================================
# Train / Validate — one epoch each
# =====================================

def run_one_epoch(model, loader, device, cross_entropy_loss, num_classes,
                   optimizer=None, scaler=None, grad_clip_norm=None):
    """Shared loop for train and validation. If optimizer is None, runs in
    eval mode with no gradient updates.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    dice_sums = [0.0] * num_classes
    n_batches = 0

    iterator = tqdm(loader, desc="Train" if is_train else "Val", leave=False) if HAS_TQDM else loader

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = model(images)
                    ce_loss = cross_entropy_loss(outputs, masks)
                    d_loss, _ = dice_loss(outputs, masks, num_classes=num_classes)
                    loss = ce_loss + d_loss
            else:
                outputs = model(images)
                ce_loss = cross_entropy_loss(outputs, masks)
                d_loss, _ = dice_loss(outputs, masks, num_classes=num_classes)
                loss = ce_loss + d_loss

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            batch_dices = per_class_dice(outputs.detach(), masks, num_classes=num_classes)
            for c in range(num_classes):
                dice_sums[c] += batch_dices[c]

            if HAS_TQDM:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_dices = [d / max(n_batches, 1) for d in dice_sums]
    return avg_loss, avg_dices


def print_dice_table(dices, num_classes):
    for c in range(num_classes):
        name = CLASS_NAMES.get(c, f"Class {c}")
        print(f"    {name:<18}: {dices[c]:.4f}")


# =====================================
# Main Training Function
# =====================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = cfg["use_amp"] and device.type == "cuda"
    if cfg["use_amp"] and device.type != "cuda":
        print("ℹ️ AMP requested but no GPU found — running in full precision.")

    # ---- Dataset ----
    dataset = BraTSDataset(cfg["data_path"], image_size=cfg["image_size"])
    if len(dataset) == 0:
        print("❌ No patients found in dataset — nothing to train on.")
        return

    val_size = max(1, int(cfg["val_split"] * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    print(f"📊 Total patients: {len(dataset)} (train={train_size}, val={val_size})")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    # ---- Model / loss / optimizer ----
    model = HybridUNetTransformer(
        in_channels=cfg["in_channels"], num_classes=cfg["num_classes"]
    ).to(device)

    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0

    # ---- Resume from checkpoint if requested ----
    if cfg["resume"] and os.path.exists(cfg["last_checkpoint_path"]):
        ckpt = torch.load(cfg["last_checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        print(f"🔄 Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_loss, train_dices = run_one_epoch(
            model, train_loader, device, cross_entropy_loss, cfg["num_classes"],
            optimizer=optimizer, scaler=scaler, grad_clip_norm=cfg["grad_clip_norm"],
        )
        print(f"Training Loss: {train_loss:.4f}")
        print("  Per-class Dice (train):")
        print_dice_table(train_dices, cfg["num_classes"])

        val_loss, val_dices = run_one_epoch(
            model, val_loader, device, cross_entropy_loss, cfg["num_classes"],
        )
        print(f"Validation Loss: {val_loss:.4f}")
        print("  Per-class Dice (val):")
        print_dice_table(val_dices, cfg["num_classes"])

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Learning rate: {current_lr:.2e}  |  Epoch time: {time.time() - epoch_start:.1f}s")

        # ---- Always save a resumable "last" checkpoint ----
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
        }, cfg["last_checkpoint_path"])

        # ---- Save best model (weights only, for easy inference loading) ----
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"✅ Best model saved! (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        # ---- Early stopping ----
        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs "
                  f"(no improvement for {cfg['early_stopping_patience']} epochs).")
            break

    print(f"\n🏁 Training finished. Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()