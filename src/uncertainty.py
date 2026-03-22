"""Uncertainty and confidence scoring."""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Tuple

from .config import N_CLASSES, UNCERTAINTY_THRESHOLD


def compute_confidence(
    proba_matrix: np.ndarray,
    n_classes: int = N_CLASSES,
    threshold: float = UNCERTAINTY_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence scores and uncertainty flags from class probabilities.

    Uses normalized entropy: confidence = 1 - H(p) / log(n_classes)
    A uniform distribution (completely unsure) gives confidence = 0.
    A peaked distribution (very sure) gives confidence -> 1.

    Args:
        proba_matrix: (n_samples, n_classes) array of class probabilities
        n_classes: Number of classes (for entropy normalization)
        threshold: Confidence threshold below which uncertain_flag = 1

    Returns:
        confidence: (n_samples,) array in [0, 1]
        uncertain_flag: (n_samples,) binary array (1 = uncertain)
    """
    if n_classes is None:
        n_classes = proba_matrix.shape[1]

    max_entropy = np.log(n_classes)

    # Compute entropy-based confidence for each sample
    confidence = np.array([
        1.0 - (scipy_entropy(p + 1e-10) / max_entropy)
        for p in proba_matrix
    ])

    confidence = np.clip(np.round(confidence, 4), 0, 1)
    uncertain_flag = (confidence < threshold).astype(int)

    return confidence, uncertain_flag


def compute_confidence_single(proba: np.ndarray, n_classes: int = N_CLASSES) -> Tuple[float, int]:
    """
    Compute confidence for a single prediction.

    Args:
        proba: (n_classes,) array of class probabilities
        n_classes: Number of classes

    Returns:
        Tuple of (confidence, uncertain_flag)
    """
    max_entropy = np.log(n_classes)
    conf = 1.0 - (scipy_entropy(proba + 1e-10) / max_entropy)
    conf = float(np.clip(np.round(conf, 4), 0, 1))
    uncertain = 1 if conf < UNCERTAINTY_THRESHOLD else 0
    return conf, uncertain
