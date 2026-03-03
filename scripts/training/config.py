"""
Shared configuration for RAGE training scripts.

CRITICAL: These values must exactly match rage-core::ExtractionConfig::default().
See: crates/rage-core/src/config.rs
"""

from pathlib import Path

# Audio parameters (match ExtractionConfig::default())
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20
FMIN = 20.0
FMAX = 11025.0
WINDOW_SECONDS = 30.0

# Derived dimensions
N_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)  # 661500
N_FRAMES = 1 + (N_SAMPLES - 1) // HOP_LENGTH  # 1292

# Model input shapes
MEL_INPUT_SHAPE = (1, N_MELS, N_FRAMES)  # [1, 128, 1292]
SUMMARY_VECTOR_DIM = 294  # 42 features x 7 statistics

# MTG-Jamendo mood/theme tags (must match rage-core::tags::MoodTag::ALL order)
MOOD_TAGS = [
    "action", "adventure", "advertising", "amusing", "angry",
    "background", "ballad", "calm", "children", "christmas",
    "commercial", "cool", "corporate", "dark", "deep",
    "documentary", "drama", "dreamy", "emotional", "energetic",
    "epic", "fast", "film", "fun", "funny",
    "game", "groovy", "happy", "heavy", "holiday",
    "hopeful", "inspiring", "love", "meditative", "melancholic",
    "melodic", "motivational", "movie", "nature", "party",
    "positive", "powerful", "relaxing", "retro", "romantic",
    "sad", "sexy", "slow", "soft", "soundscape",
    "space", "sport", "summer", "trailer", "travel",
    "uplifting",
]
NUM_MOOD_TAGS = 56

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
