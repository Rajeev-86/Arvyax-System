"""Data preprocessing functions."""

import pandas as pd
import numpy as np
from .config import DEFAULT_SLEEP_MEDIAN


def preprocess(df: pd.DataFrame, sleep_median: float = DEFAULT_SLEEP_MEDIAN) -> pd.DataFrame:
    """
    Clean and engineer features for the emotional intelligence pipeline.

    Args:
        df: Input DataFrame with raw features
        sleep_median: Value to impute for missing sleep_hours (from training set)

    Returns:
        Preprocessed DataFrame with engineered features
    """
    df = df.copy()

    # --- Missing value imputation ---
    df['sleep_hours'] = df['sleep_hours'].fillna(sleep_median)
    df['previous_day_mood'] = df['previous_day_mood'].fillna('unknown')
    df['face_emotion_hint'] = df['face_emotion_hint'].fillna('unknown')

    # --- Text cleaning ---
    df['journal_text'] = (
        df['journal_text']
        .fillna('')
        .str.lower()
        .str.strip()
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
    )

    # --- Derived text features ---
    df['text_length'] = df['journal_text'].str.len()
    df['word_count'] = df['journal_text'].str.split().str.len()
    df['is_short_text'] = (df['word_count'] <= 3).astype(int)

    # --- Interaction features ---
    df['stress_x_energy'] = df['stress_level'] * df['energy_level']

    # --- Sleep deficit flag ---
    df['sleep_deficit'] = (df['sleep_hours'] < 6).astype(int)

    return df


def preprocess_single(data: dict, sleep_median: float = DEFAULT_SLEEP_MEDIAN) -> dict:
    """
    Preprocess a single input record.

    Args:
        data: Dictionary with input features
        sleep_median: Value to impute for missing sleep_hours

    Returns:
        Preprocessed dictionary with engineered features
    """
    data = data.copy()

    # Missing value imputation
    if data.get('sleep_hours') is None:
        data['sleep_hours'] = sleep_median
    if data.get('previous_day_mood') is None or data.get('previous_day_mood') == '':
        data['previous_day_mood'] = 'unknown'
    if data.get('face_emotion_hint') is None or data.get('face_emotion_hint') == '':
        data['face_emotion_hint'] = 'unknown'

    # Text cleaning
    text = data.get('journal_text', '')
    text = str(text).lower().strip()
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    data['journal_text'] = text

    # Derived features
    data['text_length'] = len(text)
    data['word_count'] = len(text.split()) if text else 0
    data['is_short_text'] = 1 if data['word_count'] <= 3 else 0

    # Interaction features
    data['stress_x_energy'] = data['stress_level'] * data['energy_level']

    # Sleep deficit
    data['sleep_deficit'] = 1 if data['sleep_hours'] < 6 else 0

    return data
