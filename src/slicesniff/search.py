"""The SliceSniff miner: beam search over readable slice definitions.

``mine_slices`` is the whole product in one call.  It takes a classifier's
predictions -- class probabilities, or hard predictions plus confidences --
the true labels, and a dict of metadata columns, and returns a
:class:`~slicesniff.cards.SniffReport` of ranked failure cards.

Prediction-only by design: no model object, no embeddings, no framework.
Anything that can emit probabilities (PyTorch, TensorFlow, scikit-learn,
a CSV of predictions) can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cards import SniffReport, build_card
from .core import SliceStats, overall_stats, slice_stats
from .features import prediction_metadata
from .predicates import Predicate, apply_definition, generate_predicates

__all__ = ["mine_slices", "sniff_model"]

_ROW_SUM_TOL = 1e-3


@dataclass(frozen=True)
class _Prepared:
    probs: np.ndarray | None
    y_pred: np.ndarray
    confidence: np.ndarray
    top2_margin: np.ndarray | None
    correct: np.ndarray
    n: int
    n_classes: int


def _as_label_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.dtype.kind in "USO":
        _, encoded = np.unique(arr, return_inverse=True)
        return encoded.astype(np.int64)
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer class labels, got dtype {arr.dtype}")
    return arr.astype(np.int64)


def _prepare(
    probs,
    y_true,
    y_pred,
    confidence,
    class_names,
) -> _Prepared:
    y = _as_label_array(y_true, name="y_true")

    if probs is not None and (y_pred is not None or confidence is not None):
        raise ValueError("pass either probs, or y_pred together with confidence -- not both")
    if probs is None and (y_pred is None) != (confidence is None):
        raise ValueError("y_pred and confidence must be passed together")

    top2_margin = None
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"probs must be a 2-D array (n_examples, n_classes), got {p.shape}")
        n, n_classes = p.shape
        if not np.isfinite(p).all():
            raise ValueError("probs contains NaN or infinite values")
        if (p < 0).any():
            raise ValueError("probs contains negative values")
        row_sums = p.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=_ROW_SUM_TOL):
            raise ValueError(
                "each row of probs must sum to 1 "
                f"(max deviation {float(np.abs(row_sums - 1.0).max()):.4f}); "
                "pass normalized probabilities"
            )
        pred = np.argmax(p, axis=1).astype(np.int64)
        conf = p[np.arange(n), pred].copy()
        if n_classes >= 2:
            ordered = np.sort(p, axis=1)
            top2_margin = ordered[:, -1] - ordered[:, -2]
    else:
        pred = _as_label_array(y_pred, name="y_pred")
        conf = np.asarray(confidence, dtype=float)
        if conf.ndim != 1:
            raise ValueError(f"confidence must be a 1-D array, got shape {conf.shape}")
        if not np.isfinite(conf).all():
            raise ValueError("confidence contains NaN or infinite values")
        if (conf < 0).any() or (conf > 1).any():
            raise ValueError("confidence values must lie in [0, 1]")
        n = pred.shape[0]
        n_classes = int(max(pred.max(initial=0), y.max(initial=0))) + 1
        p = None

    if y.shape[0] != n:
        raise ValueError(f"y_true has length {y.shape[0]}, expected {n}")
    if n == 0:
        raise ValueError("empty dataset: no examples to audit")
    if y.min() < 0 or pred.min() < 0:
        raise ValueError("class labels must be non-negative")
    if y.max() >= n_classes or pred.max() >= n_classes:
        raise ValueError(
            f"labels reach class {int(max(y.max(), pred.max()))} but probs imply "
            f"{n_classes} classes"
        )
    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but the data imply {n_classes} classes"
        )
    return _Prepared(p, pred, conf, top2_margin, pred == y, n, n_classes)


def mine_slices(
    probs: np.ndarray | None = None,
    y_true=None,
    metadata: dict | None = None,
    *,
    y_pred=None,
    confidence=None,
    class_names: list | tuple | None = None,
    min_support: int = 30,
    min_slice_fraction: float = 0.01,
    max_depth: int = 2,
    beam_width: int = 25,
    top_k: int = 10,
    min_gap: float = 0.15,
    min_score: float = 0.0,
    include_prediction_features: bool = False,
    overlap_threshold: float = 0.9,
    max_evaluated: int = 20000,
    ece_bins: int = 10,
) -> SniffReport:
    """Find the slices where a classifier is confidently wrong.

    Give it predictions and metadata; get back ranked, reproducible
    failure cards.

    Parameters
    ----------
    probs:
        ``(n, n_classes)`` array of predicted class probabilities.  Rows
        must sum to 1.  Mutually exclusive with ``y_pred``/``confidence``.
    y_true:
        True labels: integer class ids, or strings (encoded internally;
        ``class_names`` then names the sorted unique values).
    metadata:
        Dict of per-example columns, each of length ``n``: categorical
        (strings/bools/ints) or numeric (auto-binned by quantiles).
        ``None`` means only the model's own prediction features are used.
    y_pred, confidence:
        Alternative to ``probs``: hard predicted labels plus the
        predicted-class probability for each example, in ``[0, 1]``.
    min_support:
        A slice must contain at least this many examples (and at least
        ``min_slice_fraction`` of the data).
    max_depth:
        Maximum number of predicates per slice definition (1 or 2 is
        usually right; deeper slices are harder to act on).
    min_gap:
        A slice only becomes a card if its overconfidence gap
        (mean confidence - accuracy) is at least this large.
    include_prediction_features:
        Also slice over columns derived from the model's own output
        (``predicted_class``, ``confidence_band``, ``margin``).  Off by
        default: those columns describe *that* the model fails when it
        is confident, which is true of almost every bad slice and would
        outrank the metadata slices that say *where* it fails.  Turn it
        on for behaviour slices; leave it off for world slices.
    overlap_threshold:
        Near-duplicate slices are dropped: a kept card suppresses a
        later candidate whose example set has Jaccard similarity above
        this threshold with it.

    Returns
    -------
    SniffReport
        Ranked failure cards (worst first) plus overall baselines and the
        exact search configuration, for reproduction.
    """
    if min_support < 2:
        raise ValueError("min_support must be at least 2")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    if min_score < 0:
        raise ValueError("min_score must be non-negative")
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must lie in (0, 1]")
    if max_evaluated < 1:
        raise ValueError("max_evaluated must be at least 1")

    prepared = _prepare(probs, y_true, y_pred, confidence, class_names)
    y = _as_label_array(y_true, name="y_true")
    n = prepared.n

    columns: dict[str, np.ndarray] = {}
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict of column name -> array")
        for name, values in metadata.items():
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"metadata column {name!r} must be a 1-D array of length {n}, "
                    f"got shape {arr.shape}"
                )
            columns[str(name)] = arr
    if include_prediction_features:
        for name, values in prediction_metadata(
            prepared.y_pred,
            prepared.confidence,
            class_names=class_names,
            top2_margin=prepared.top2_margin,
        ).items():
            columns.setdefault(name, values)
    # Note: y_true is deliberately NOT offered as a slice feature.  A
    # definition mentioning both the true label and the prediction (e.g.
    # "label == 0 AND predicted_class == 2") is wrong by construction and
    # would flood the ranking with tautological confusion pairs.  Slices
    # describe the input and the model's behaviour; the label only
    # defines correctness.

    min_count = max(int(min_support), int(np.ceil(min_slice_fraction * n)))
    overall = overall_stats(prepared.correct, prepared.confidence, ece_bins=ece_bins)

    by_feature: dict[str, list[Predicate]] = {}
    for feature in sorted(columns):
        generated = generate_predicates(
            feature, columns[feature], n=n, min_support=min_count
        )
        if generated:
            by_feature[feature] = generated
    all_predicates = [p for feature in sorted(by_feature) for p in by_feature[feature]]

    config = {
        "min_support": int(min_support),
        "min_slice_fraction": float(min_slice_fraction),
        "effective_min_count": int(min_count),
        "max_depth": int(max_depth),
        "beam_width": int(beam_width),
        "top_k": int(top_k),
        "min_gap": float(min_gap),
        "min_score": float(min_score),
        "min_ece": 0.0,
        "include_prediction_features": bool(include_prediction_features),
        "overlap_threshold": float(overlap_threshold),
        "max_evaluated": int(max_evaluated),
        "ece_bins": int(ece_bins),
        "features_searched": sorted(by_feature),
    }

    evaluated: dict[tuple[Predicate, ...], tuple[np.ndarray, SliceStats]] = {}
    evaluated_total = 0

    def consider(definition: tuple[Predicate, ...], mask: np.ndarray):
        nonlocal evaluated_total
        if definition in evaluated or evaluated_total >= max_evaluated:
            return None
        stats = slice_stats(
            prepared.correct, prepared.confidence, np.flatnonzero(mask), ece_bins=ece_bins
        )
        evaluated[definition] = (mask, stats)
        evaluated_total += 1
        return stats

    for predicate in all_predicates:
        if evaluated_total >= max_evaluated:
            break
        consider((predicate,), predicate.mask)

    def rank_key(definition: tuple[Predicate, ...]):
        stats = evaluated[definition][1]
        return (-stats.score, -stats.gap, -stats.size, tuple(p.text for p in definition))

    parents = sorted(evaluated, key=rank_key)[:beam_width]
    depth = 1
    while depth < max_depth and parents and evaluated_total < max_evaluated:
        children: list[tuple[Predicate, ...]] = []
        for parent in parents:
            used = {p.feature for p in parent}
            base_mask = evaluated[parent][0]
            for feature in sorted(by_feature):
                if feature in used:
                    continue
                for predicate in by_feature[feature]:
                    mask = base_mask & predicate.mask
                    if int(mask.sum()) < min_count:
                        continue
                    child = tuple(sorted(parent + (predicate,), key=lambda p: p.text))
                    if consider(child, mask) is not None:
                        children.append(child)
                    if evaluated_total >= max_evaluated:
                        break
                if evaluated_total >= max_evaluated:
                    break
            if evaluated_total >= max_evaluated:
                break
        parents = sorted(set(children), key=rank_key)[:beam_width]
        depth += 1

    def passes(stats: SliceStats) -> bool:
        return (
            stats.gap >= min_gap
            and stats.accuracy < 1.0
            and (min_score <= 0.0 or stats.score >= min_score)
        )

    kept: list[tuple[tuple[Predicate, ...], np.ndarray, SliceStats]] = []
    kept_sets: list[set[int]] = []
    for definition in sorted(evaluated, key=rank_key):
        if len(kept) >= top_k:
            break
        mask, stats = evaluated[definition]
        if not passes(stats):
            continue
        indices = set(int(i) for i in np.flatnonzero(mask))
        duplicate = False
        for other in kept_sets:
            union = len(indices | other)
            if union and len(indices & other) / union > overlap_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((definition, mask, stats))
        kept_sets.append(indices)

    cards = [
        build_card(
            definition,
            np.flatnonzero(mask),
            stats,
            overall=overall,
            dataset_size=n,
            y_true=y,
            y_pred=prepared.y_pred,
            class_names=class_names,
        )
        for definition, mask, stats in kept
    ]
    return SniffReport(
        cards=cards,
        overall=overall,
        dataset_size=n,
        n_classes=prepared.n_classes,
        config=config,
    )


def sniff_model(model, X, y_true, metadata: dict | None = None, **kwargs) -> SniffReport:
    """Mine slices for a fitted model that exposes ``predict_proba``.

    Works with anything scikit-learn-shaped (including pipelines) -- and
    with wrappers around other frameworks that offer the same method.
    SliceSniff itself depends only on NumPy; the model is yours.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "sniff_model needs a model with predict_proba; "
            "otherwise compute probabilities yourself and call mine_slices"
        )
    probs = np.asarray(model.predict_proba(X), dtype=float)
    return mine_slices(probs, y_true, metadata, **kwargs)
