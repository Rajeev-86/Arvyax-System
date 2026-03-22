"""Feature engineering: embeddings and feature matrix construction."""

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    METADATA_COLS, CATEGORICAL_COLS,
    SCALER_PATH, ENCODERS_PATH
)


class FeatureEngineer:
    """Handles text embedding and feature matrix construction."""

    def __init__(
        self,
        scaler_path: Path = SCALER_PATH,
        encoders_path: Path = ENCODERS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Initialize feature engineer with pre-trained artifacts.

        Args:
            scaler_path: Path to saved StandardScaler
            encoders_path: Path to saved LabelEncoders dict
            embedding_model_name: Sentence transformer model name
        """
        # Load pre-trained artifacts
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded.")

    def encode_text(self, texts: list) -> np.ndarray:
        """
        Encode texts to embeddings using sentence transformer.

        Args:
            texts: List of text strings

        Returns:
            (n_samples, embedding_dim) array
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True
        )
        return embeddings

    def encode_text_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        embedding = self.embedding_model.encode(
            [text],
            normalize_embeddings=True
        )
        return embedding[0]

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.

        Args:
            df: DataFrame with categorical columns

        Returns:
            DataFrame with encoded columns added
        """
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                encoder = self.encoders.get(col)
                if encoder is not None:
                    # Handle unseen categories
                    df[col + '_enc'] = df[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0]
                        if x in encoder.classes_
                        else 0  # Default to first class for unknown
                    )
                else:
                    df[col + '_enc'] = 0
        return df

    def encode_categorical_single(self, data: dict) -> dict:
        """Apply label encoding to a single record."""
        data = data.copy()
        for col in CATEGORICAL_COLS:
            if col in data:
                encoder = self.encoders.get(col)
                if encoder is not None:
                    val = str(data[col])
                    if val in encoder.classes_:
                        data[col + '_enc'] = int(encoder.transform([val])[0])
                    else:
                        data[col + '_enc'] = 0
                else:
                    data[col + '_enc'] = 0
        return data

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Combine embeddings and metadata into full feature matrix.

        Args:
            df: Preprocessed DataFrame with encoded categorical cols
            embeddings: Text embeddings array

        Returns:
            Full feature matrix (n_samples, embedding_dim + n_metadata)
        """
        # Extract metadata features
        X_meta = df[METADATA_COLS].values.astype(float)

        # Scale metadata
        X_meta_scaled = self.scaler.transform(X_meta)

        # Combine embeddings and metadata
        X_full = np.hstack([embeddings, X_meta_scaled])
        return X_full

    def build_feature_vector(self, data: dict, embedding: np.ndarray) -> np.ndarray:
        """
        Build feature vector for a single sample.

        Args:
            data: Preprocessed dict with encoded categorical cols
            embedding: Text embedding (1D array)

        Returns:
            Feature vector (1D array)
        """
        # Extract metadata in correct order
        meta_values = [float(data.get(col, 0)) for col in METADATA_COLS]
        X_meta = np.array(meta_values).reshape(1, -1)

        # Scale
        X_meta_scaled = self.scaler.transform(X_meta)

        # Combine
        X_full = np.hstack([embedding.reshape(1, -1), X_meta_scaled])
        return X_full[0]


def create_feature_engineer(
    scaler_path: Optional[Path] = None,
    encoders_path: Optional[Path] = None
) -> FeatureEngineer:
    """Factory function to create FeatureEngineer."""
    return FeatureEngineer(
        scaler_path=scaler_path or SCALER_PATH,
        encoders_path=encoders_path or ENCODERS_PATH
    )
