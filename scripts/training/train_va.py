"""
Train the ValenceArousalModel on DEAM dataset.

Usage:
    python scripts/training/train_va.py \
        --data-dir data/deam \
        --epochs 80 \
        --batch-size 16 \
        --lr 3e-4

Outputs:
    models/checkpoints/va_best.pt
    models/checkpoints/va_final.pt
    TensorBoard logs in models/runs/valence_arousal/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINTS_DIR, MODELS_DIR
from models.losses import CombinedVALoss
from models.valence_arousal import ValenceArousalModel
from utils.augment import SpecAugment
from utils.data import DEAMDataset
from utils.device import get_device
from utils.metrics import concordance_correlation_coefficient


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mel, summary, va_target in loader:
        mel = mel.to(device)
        summary = summary.to(device)
        va_target = va_target.to(device)

        optimizer.zero_grad()
        va_pred = model(mel, summary)
        loss = criterion(va_pred, va_target)
        loss.backward()

        # Gradient clipping (important for CCC loss stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for mel, summary, va_target in loader:
        mel = mel.to(device)
        summary = summary.to(device)
        va_target = va_target.to(device)

        va_pred = model(mel, summary)
        loss = criterion(va_pred, va_target)
        total_loss += loss.item()

        all_preds.append(va_pred.cpu().numpy())
        all_targets.append(va_target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ccc_v = concordance_correlation_coefficient(all_preds[:, 0], all_targets[:, 0])
    ccc_a = concordance_correlation_coefficient(all_preds[:, 1], all_targets[:, 1])
    ccc_mean = (ccc_v + ccc_a) / 2.0

    return total_loss / max(len(loader), 1), ccc_v, ccc_a, ccc_mean


def main():
    parser = argparse.ArgumentParser(description="Train ValenceArousalModel")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)

    # Data
    data_dir = Path(args.data_dir)
    augment = SpecAugment(
        freq_mask_param=15,
        time_mask_param=60,
        num_freq_masks=2,
        num_time_masks=2,
    )

    train_ds = DEAMDataset(
        split_file=str(data_dir / "splits" / "train.tsv"),
        features_dir=str(data_dir / "features"),
        transform=augment,
    )
    val_ds = DEAMDataset(
        split_file=str(data_dir / "splits" / "val.tsv"),
        features_dir=str(data_dir / "features"),
        transform=None,
    )

    pin_memory = device.type not in ("mps",)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = ValenceArousalModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ValenceArousalModel parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = CombinedVALoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Logging
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(MODELS_DIR / "runs" / "valence_arousal"))

    # Training loop
    best_ccc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, ccc_v, ccc_a, ccc_mean = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"CCC_v={ccc_v:.4f} | CCC_a={ccc_a:.4f} | "
            f"CCC_mean={ccc_mean:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("ccc/valence", ccc_v, epoch)
        writer.add_scalar("ccc/arousal", ccc_a, epoch)
        writer.add_scalar("ccc/mean", ccc_mean, epoch)
        writer.add_scalar("lr", lr, epoch)

        if ccc_mean > best_ccc:
            best_ccc = ccc_mean
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ccc_valence": ccc_v,
                    "ccc_arousal": ccc_a,
                    "ccc_mean": ccc_mean,
                },
                CHECKPOINTS_DIR / "va_best.pt",
            )
            print(f"  -> New best CCC: {best_ccc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={args.patience})"
                )
                break

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "va_final.pt")
    writer.close()
    print(f"Training complete. Best CCC: {best_ccc:.4f}")


if __name__ == "__main__":
    main()
