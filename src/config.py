"""Configuration constants for the ArvyaX pipeline."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Model file paths
CLF_STATE_PATH = MODELS_DIR / "clf_emotional_state.pkl"
REG_INTENSITY_PATH = MODELS_DIR / "reg_intensity.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
STATE_ENCODER_PATH = MODELS_DIR / "state_encoder.pkl"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# SLM for supportive messages
SLM_MODEL_NAME = "google/gemma-3-1b-it"  # Will use gemma-3-270m if available

# Feature columns
METADATA_COLS = [
    'duration_min', 'sleep_hours', 'energy_level', 'stress_level',
    'stress_x_energy', 'sleep_deficit',
    'text_length', 'word_count', 'is_short_text',
    'ambience_type_enc', 'time_of_day_enc', 'previous_day_mood_enc',
    'face_emotion_hint_enc', 'reflection_quality_enc'
]

CATEGORICAL_COLS = [
    'ambience_type', 'time_of_day', 'previous_day_mood',
    'face_emotion_hint', 'reflection_quality'
]

# Emotional states (from training)
EMOTIONAL_STATES = ['calm', 'focused', 'mixed', 'neutral', 'overwhelmed', 'restless']
N_CLASSES = len(EMOTIONAL_STATES)

# Default imputation values (from training set medians)
DEFAULT_SLEEP_MEDIAN = 7.0

# Uncertainty threshold
UNCERTAINTY_THRESHOLD = 0.55

# Random seed
SEED = 42
