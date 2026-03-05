"""
Train the MoodTagger CNN on MTG-Jamendo mood/theme tags.

Usage:
    python scripts/training/train_mood.py \
        --data-dir data/jamendo \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3

Outputs:
    models/checkpoints/mood_tagger_best.pt
    models/checkpoints/mood_tagger_final.pt
    TensorBoard logs in models/runs/mood_tagger/
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINTS_DIR, MODELS_DIR, NUM_MOOD_TAGS
from models.mood_tagger import MoodTagger
from utils.augment import SpecAugment
from utils.data import JamendoMoodDataset
from utils.device import get_device
from utils.metrics import compute_map, compute_roc_auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mel, labels in loader:
        mel = mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for mel, labels in loader:
        mel = mel.to(device)
        labels = labels.to(device)

        logits = model(mel)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)
    mAP = compute_map(all_labels.numpy(), probs.numpy())
    roc_auc = compute_roc_auc(all_labels.numpy(), probs.numpy())

    return total_loss / max(len(loader), 1), mAP, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Train MoodTagger")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--mel-dir", type=str, default=None,
                        help="Override mel spectrogram directory (default: <data-dir>/mel_spectrograms)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)

    # Data
    data_dir = Path(args.data_dir)
    mel_dir = Path(args.mel_dir) if args.mel_dir else data_dir / "mel_spectrograms"
    print(f"Mel spectrogram directory: {mel_dir}")

    augment = SpecAugment(
        freq_mask_param=20,
        time_mask_param=80,
        num_freq_masks=2,
        num_time_masks=2,
    )

    train_ds = JamendoMoodDataset(
        split_file=str(data_dir / "splits" / "train_processed.tsv"),
        mel_dir=str(mel_dir),
        transform=augment,
    )
    val_ds = JamendoMoodDataset(
        split_file=str(data_dir / "splits" / "val_processed.tsv"),
        mel_dir=str(mel_dir),
        transform=None,
    )

    # pin_memory=True is not supported with MPS
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
    model = MoodTagger(num_tags=NUM_MOOD_TAGS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MoodTagger parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Logging
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(MODELS_DIR / "runs" / "mood_tagger"))

    # Training loop
    best_mAP = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mAP, val_roc_auc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mAP={val_mAP:.4f} | val_AUC={val_roc_auc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/mAP", val_mAP, epoch)
        writer.add_scalar("metrics/roc_auc", val_roc_auc, epoch)
        writer.add_scalar("lr", lr, epoch)

        if val_mAP > best_mAP:
            best_mAP = val_mAP
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mAP": val_mAP,
                    "val_roc_auc": val_roc_auc,
                },
                CHECKPOINTS_DIR / "mood_tagger_best.pt",
            )
            print(f"  -> New best mAP: {best_mAP:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={args.patience})"
                )
                break

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "mood_tagger_final.pt")
    writer.close()
    print(f"Training complete. Best mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    main()
