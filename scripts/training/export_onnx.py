"""
Export trained PyTorch models to ONNX format.

CRITICAL: Must export on CPU, not MPS.
PyTorch ONNX export produces incorrect results for Conv2d+BatchNorm
models when run on MPS. This is a known issue.

Usage:
    python scripts/training/export_onnx.py \
        --mood-checkpoint models/checkpoints/mood_tagger_best.pt \
        --va-checkpoint models/checkpoints/va_best.pt \
        --output-dir models/
"""

import argparse
import sys
from pathlib import Path

import onnx
import torch
import torch.onnx

sys.path.insert(0, str(Path(__file__).parent))

from config import N_FRAMES, N_MELS, NUM_MOOD_TAGS, SUMMARY_VECTOR_DIM
from models.mood_tagger import MoodTagger
from models.valence_arousal import ValenceArousalModel

OPSET_VERSION = 16


def export_mood_tagger(checkpoint_path: str, output_path: str):
    """Export MoodTagger to ONNX."""
    print(f"Exporting MoodTagger from {checkpoint_path}")

    # CRITICAL: Load and export on CPU only
    device = torch.device("cpu")

    model = MoodTagger(num_tags=NUM_MOOD_TAGS)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    dummy_mel = torch.randn(1, 1, N_MELS, N_FRAMES, device=device)

    torch.onnx.export(
        model,
        dummy_mel,
        output_path,
        opset_version=OPSET_VERSION,
        input_names=["mel_spectrogram"],
        output_names=["mood_logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "mood_logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"  Exported to {output_path}")
    print(f"  ONNX opset: {OPSET_VERSION}")
    print(f"  Input shape: [batch, 1, {N_MELS}, {N_FRAMES}]")
    print(f"  Output shape: [batch, {NUM_MOOD_TAGS}]")


def export_valence_arousal(checkpoint_path: str, output_path: str):
    """Export ValenceArousalModel to ONNX.

    This model has TWO inputs (mel + summary vector).
    """
    print(f"Exporting ValenceArousalModel from {checkpoint_path}")

    device = torch.device("cpu")

    model = ValenceArousalModel()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    dummy_mel = torch.randn(1, 1, N_MELS, N_FRAMES, device=device)
    dummy_summary = torch.randn(1, SUMMARY_VECTOR_DIM, device=device)

    torch.onnx.export(
        model,
        (dummy_mel, dummy_summary),
        output_path,
        opset_version=OPSET_VERSION,
        input_names=["mel_spectrogram", "summary_vector"],
        output_names=["valence_arousal"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "summary_vector": {0: "batch_size"},
            "valence_arousal": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"  Exported to {output_path}")
    print(f"  ONNX opset: {OPSET_VERSION}")
    print(f"  Input 1: mel_spectrogram [batch, 1, {N_MELS}, {N_FRAMES}]")
    print(f"  Input 2: summary_vector [batch, {SUMMARY_VECTOR_DIM}]")
    print(f"  Output: valence_arousal [batch, 2]")


def main():
    parser = argparse.ArgumentParser(
        description="Export RAGE models to ONNX"
    )
    parser.add_argument("--mood-checkpoint", type=str, required=True)
    parser.add_argument("--va-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_mood_tagger(
        args.mood_checkpoint, str(output_dir / "mood_tagger.onnx")
    )
    print()
    export_valence_arousal(
        args.va_checkpoint, str(output_dir / "valence_arousal.onnx")
    )

    print("\nONNX export complete.")
    print("Next: run verify_onnx.py to compare PyTorch vs ONNX outputs.")


if __name__ == "__main__":
    main()
