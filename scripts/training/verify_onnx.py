"""
Verify ONNX models produce identical outputs to PyTorch models.

Loads both the PyTorch checkpoint and the exported ONNX file,
runs the same random input through both, and compares outputs.

Both PyTorch and ONNX inference run on CPU for fair comparison.

Usage:
    python scripts/training/verify_onnx.py \
        --mood-checkpoint models/checkpoints/mood_tagger_best.pt \
        --mood-onnx models/mood_tagger.onnx \
        --va-checkpoint models/checkpoints/va_best.pt \
        --va-onnx models/valence_arousal.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import N_FRAMES, N_MELS, NUM_MOOD_TAGS, SUMMARY_VECTOR_DIM
from models.mood_tagger import MoodTagger
from models.valence_arousal import ValenceArousalModel


def verify_mood_tagger(
    checkpoint_path: str, onnx_path: str, atol: float = 1e-5
) -> bool:
    """Compare MoodTagger PyTorch vs ONNX outputs."""
    print("=== Verifying MoodTagger ===")

    device = torch.device("cpu")
    model = MoodTagger(num_tags=NUM_MOOD_TAGS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    session = ort.InferenceSession(onnx_path)

    np.random.seed(42)
    max_abs_diff = 0.0

    for i in range(5):
        dummy = np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(dummy)).numpy()

        onnx_out = session.run(None, {"mel_spectrogram": dummy})[0]

        diff = np.abs(pt_out - onnx_out).max()
        max_abs_diff = max(max_abs_diff, diff)
        print(f"  Test {i + 1}: max_abs_diff = {diff:.2e}")

    passed = max_abs_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status} (max_diff={max_abs_diff:.2e}, tol={atol:.2e})")
    return passed


def verify_va_model(
    checkpoint_path: str, onnx_path: str, atol: float = 1e-5
) -> bool:
    """Compare ValenceArousalModel PyTorch vs ONNX outputs."""
    print("\n=== Verifying ValenceArousalModel ===")

    device = torch.device("cpu")
    model = ValenceArousalModel()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    session = ort.InferenceSession(onnx_path)

    np.random.seed(42)
    max_abs_diff = 0.0

    for i in range(5):
        dummy_mel = np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)
        dummy_summary = np.random.randn(1, SUMMARY_VECTOR_DIM).astype(np.float32)

        with torch.no_grad():
            pt_out = model(
                torch.from_numpy(dummy_mel),
                torch.from_numpy(dummy_summary),
            ).numpy()

        onnx_out = session.run(
            None,
            {
                "mel_spectrogram": dummy_mel,
                "summary_vector": dummy_summary,
            },
        )[0]

        diff = np.abs(pt_out - onnx_out).max()
        max_abs_diff = max(max_abs_diff, diff)
        print(f"  Test {i + 1}: max_abs_diff = {diff:.2e}")

    passed = max_abs_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status} (max_diff={max_abs_diff:.2e}, tol={atol:.2e})")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify ONNX model parity with PyTorch"
    )
    parser.add_argument("--mood-checkpoint", type=str, required=True)
    parser.add_argument("--mood-onnx", type=str, required=True)
    parser.add_argument("--va-checkpoint", type=str, required=True)
    parser.add_argument("--va-onnx", type=str, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    mood_ok = verify_mood_tagger(
        args.mood_checkpoint, args.mood_onnx, atol=args.atol
    )
    va_ok = verify_va_model(
        args.va_checkpoint, args.va_onnx, atol=args.atol
    )

    if mood_ok and va_ok:
        print("\nAll verifications PASSED.")
    else:
        print("\nSome verifications FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
