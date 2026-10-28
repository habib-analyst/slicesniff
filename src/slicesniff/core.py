"""Core data types and statistics for SliceSniff.

A slice is a subset of examples selected by a conjunction of human-readable
predicates over metadata columns.  For every candidate slice SliceSniff
measures how the classifier behaved inside it:

- ``accuracy``        -- fraction of examples whose predicted class is right
- ``mean_confidence`` -- mean probability the model assigned to its own
                         predicted class
- ``gap``             -- ``mean_confidence - accuracy``: the overconfidence
                         gap.  A slice can have mediocre accuracy and still
                         be honest (low confidence); the dangerous slices
                         are the ones where the gap is large.
- ``ece``             -- expected calibration error inside the slice
                         (confidence-binned |confidence - accuracy|)

The ranking score is ``gap * log1p(size)``: a large gap on a large slice
outranks the same gap on a handful of examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["SliceStats", "slice_stats", "overall_stats"]


@dataclass(frozen=True)
class SliceStats:
    """Behaviour of a classifier on one subset of examples."""

    size: int
    errors: int
    accuracy: float
    mean_confidence: float
    gap: float
    ece: float
    score: float


def slice_stats(
    correct: np.ndarray,
    confidence: np.ndarray,
    indices: np.ndarray,
    *,
    ece_bins: int = 10,
) -> SliceStats:
    """Compute :class:`SliceStats` for the examples at ``indices``.

    ``correct`` is a boolean array (predicted class == true class) and
    ``confidence`` the predicted-class probability, both of length N.
    """
    if indices.size == 0:
        raise ValueError("cannot compute stats for an empty slice")
    corr = correct[indices]
    conf = confidence[indices]
    size = int(indices.size)
    accuracy = float(corr.mean())
    mean_confidence = float(conf.mean())
    gap = mean_confidence - accuracy
    # Equal-width ECE bins over [0, 1].
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges[1:-1]), 0, ece_bins - 1)
    ece = 0.0
    for b in range(ece_bins):
        in_bin = bin_id == b
        count = int(in_bin.sum())
        if count:
            ece += (count / size) * abs(float(conf[in_bin].mean()) - float(corr[in_bin].mean()))
    return SliceStats(
        size=size,
        errors=int(size - corr.sum()),
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        gap=float(gap),
        ece=float(ece),
        score=float(gap * math.log1p(size)),
    )


def overall_stats(
    correct: np.ndarray,
    confidence: np.ndarray,
    *,
    ece_bins: int = 10,
) -> SliceStats:
    """Stats for the full dataset (the baseline every slice is compared to)."""
    return slice_stats(correct, confidence, np.arange(correct.shape[0]), ece_bins=ece_bins)
