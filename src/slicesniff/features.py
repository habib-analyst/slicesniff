"""Optional helpers: derive metadata columns from raw inputs.

SliceSniff's core needs nothing but predictions and a metadata dict, so
these helpers are conveniences, not requirements.  ``image_stats`` turns a
batch of images into the simple per-image statistics that make good slice
features (brightness band, contrast, border-vs-center ratio, size), and
``prediction_metadata`` exposes the model's own output (predicted class,
confidence band, margin) as sliceable columns.
"""

from __future__ import annotations

import numpy as np

__all__ = ["image_stats", "prediction_metadata"]

_EPS = 1e-6


def image_stats(images: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image statistics for a batch of images.

    Accepts ``(N, H, W)`` or ``(N, H, W, C)`` arrays of any numeric dtype
    and returns global per-image statistics (computed over all pixels and
    channels): ``brightness`` (mean), ``contrast`` (std), ``edge_density``
    (mean absolute neighbour difference), ``border_center_ratio`` (mean of
    the outer frame divided by the mean of the centre crop), ``height``
    and ``width``.
    """
    x = np.asarray(images)
    if x.ndim not in (3, 4):
        raise ValueError(f"expected images with shape (N, H, W) or (N, H, W, C), got {x.shape}")
    n, height, width = x.shape[0], x.shape[1], x.shape[2]
    xf = x.astype(np.float64)
    flat = xf.reshape(n, -1)
    brightness = flat.mean(axis=1)
    contrast = flat.std(axis=1)

    diffs = []
    if height > 1:
        diffs.append(np.abs(np.diff(xf, axis=1)).reshape(n, -1))
    if width > 1:
        diffs.append(np.abs(np.diff(xf, axis=2)).reshape(n, -1))
    if diffs:
        edge_density = np.concatenate(diffs, axis=1).mean(axis=1)
    else:  # pragma: no cover - degenerate 1x1 images
        edge_density = np.zeros(n)

    h0, h1 = height // 4, max(height // 4 + 1, (3 * height) // 4)
    w0, w1 = width // 4, max(width // 4 + 1, (3 * width) // 4)
    centre = xf[:, h0:h1, w0:w1].reshape(n, -1)
    centre_sum = centre.sum(axis=1)
    centre_count = centre.shape[1]
    total_sum = flat.sum(axis=1)
    total_count = flat.shape[1]
    border_count = max(total_count - centre_count, 1)
    border_mean = (total_sum - centre_sum) / border_count
    centre_mean = centre_sum / centre_count
    ratio = border_mean / (np.abs(centre_mean) + 1e-3)
    border_center_ratio = np.clip(ratio, 0.0, 10.0)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "border_center_ratio": border_center_ratio,
        "height": np.full(n, float(height)),
        "width": np.full(n, float(width)),
    }


def prediction_metadata(
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    class_names: list | tuple | None = None,
    top2_margin: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sliceable columns derived from the model's own predictions.

    Returns ``predicted_class`` (names if ``class_names`` is given),
    ``confidence_band`` (``low`` / ``mid`` / ``high`` / ``very_high``) and,
    if ``top2_margin`` (top1 - top2 probability) is supplied, ``margin``.
    """
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    if class_names is not None:
        names = list(class_names)
        predicted = np.array([names[int(k)] for k in y_pred], dtype=object)
    else:
        # Object dtype on purpose: predicted classes are categories, and
        # must generate "predicted_class == 2" predicates, not numeric
        # bands like "0 <= predicted_class <= 1".
        predicted = y_pred.astype(object)
    bands = np.where(
        confidence < 0.5,
        "low",
        np.where(confidence < 0.75, "mid", np.where(confidence < 0.9, "high", "very_high")),
    )
    metadata: dict[str, np.ndarray] = {
        "predicted_class": predicted,
        "confidence_band": bands.astype(object),
    }
    if top2_margin is not None:
        metadata["margin"] = np.asarray(top2_margin, dtype=float)
    return metadata
