"""SliceSniff -- sniff out the slices where a classifier is confidently wrong.

Hand it a classifier's predictions (probabilities + true labels + a dict
of metadata columns) and it mines human-readable slices -- conjunctions
of predicates over metadata and image statistics -- where accuracy
collapses while confidence stays high, and returns ranked, reproducible
failure cards.

>>> import numpy as np
>>> from slicesniff import mine_slices
>>> report = mine_slices(probs, y_true, {"source": sources, "brightness": brightness})
>>> print(report.cards[0].summary())

Prediction-only: no model object, no embeddings, no framework.  The only
dependency is NumPy.
"""

from .cards import FailureCard, SniffReport
from .core import SliceStats, overall_stats, slice_stats
from .features import image_stats, prediction_metadata
from .predicates import Predicate
from .search import mine_slices, sniff_model

__version__ = "0.1.0"

__all__ = [
    "FailureCard",
    "Predicate",
    "SliceStats",
    "SniffReport",
    "image_stats",
    "mine_slices",
    "overall_stats",
    "prediction_metadata",
    "slice_stats",
    "sniff_model",
    "__version__",
]
