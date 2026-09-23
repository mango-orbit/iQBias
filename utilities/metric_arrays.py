"""Lightweight array helpers used by metric evaluation."""

import numpy as np


def concatenate_feature_batches(batches):
    """Join CPU feature batches without materializing nested Python lists."""
    if not batches:
        raise ValueError("No feature batches were collected.")
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)
