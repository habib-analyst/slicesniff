"""Failure cards: the reproducible output artifact of SliceSniff.

A card is everything needed to re-find and re-check one bad slice without
SliceSniff in the loop: the readable slice definition, the indices of the
examples in the slice (in the caller's row order), the behaviour numbers
(size, errors, accuracy, mean confidence, the gap, in-slice ECE), the
overall-dataset baselines the slice was compared against, the dominant
confusion inside the slice, and a copy of the search configuration that
produced it.  Cards serialize to plain JSON-safe dicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .core import SliceStats
from .predicates import Predicate, definition_text

__all__ = ["FailureCard", "SniffReport"]


@dataclass
class FailureCard:
    """One confidently-wrong slice, fully described."""

    definition: str
    predicates: tuple[str, ...]
    indices: tuple[int, ...]
    size: int
    dataset_size: int
    errors: int
    accuracy: float
    mean_confidence: float
    gap: float
    ece: float
    score: float
    error_rate: float = field(init=False)
    coverage: float = field(init=False)
    overall_accuracy: float = 0.0
    overall_mean_confidence: float = 0.0
    top_confusion: tuple[str, str, int] | None = None

    def __post_init__(self) -> None:
        self.error_rate = 1.0 - self.accuracy
        self.coverage = self.size / self.dataset_size if self.dataset_size else 0.0

    def to_dict(self) -> dict:
        return {
            "definition": self.definition,
            "predicates": list(self.predicates),
            "indices": [int(i) for i in self.indices],
            "size": self.size,
            "dataset_size": self.dataset_size,
            "coverage": self.coverage,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "gap": self.gap,
            "ece": self.ece,
            "score": self.score,
            "overall_accuracy": self.overall_accuracy,
            "overall_mean_confidence": self.overall_mean_confidence,
            "top_confusion": (
                None
                if self.top_confusion is None
                else {
                    "true": self.top_confusion[0],
                    "predicted": self.top_confusion[1],
                    "count": self.top_confusion[2],
                }
            ),
        }

    def summary(self) -> str:
        lines = [
            f"slice:       {self.definition}",
            f"size:        {self.size} of {self.dataset_size} examples "
            f"({self.coverage:.1%} of the data, indices in 'indices')",
            f"accuracy:    {self.accuracy:.3f}  (overall {self.overall_accuracy:.3f})",
            f"confidence:  {self.mean_confidence:.3f}  (overall {self.overall_mean_confidence:.3f})",
            f"gap:         {self.gap:+.3f}  (confidence - accuracy; positive = overconfident)",
            f"errors:      {self.errors} of {self.size} wrong, in-slice ECE {self.ece:.3f}",
        ]
        if self.top_confusion is not None:
            true_name, pred_name, count = self.top_confusion
            lines.append(
                f"top error:   true '{true_name}' predicted as '{pred_name}' x{count}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class SniffReport:
    """The full result of one :func:`slicesniff.mine_slices` run."""

    cards: list[FailureCard]
    overall: SliceStats
    dataset_size: int
    n_classes: int
    config: dict

    def to_dict(self) -> dict:
        return {
            "dataset_size": self.dataset_size,
            "n_classes": self.n_classes,
            "overall": {
                "accuracy": self.overall.accuracy,
                "mean_confidence": self.overall.mean_confidence,
                "gap": self.overall.gap,
                "ece": self.overall.ece,
            },
            "config": self.config,
            "cards": [card.to_dict() for card in self.cards],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        header = [
            "=" * 68,
            "SLICESNIFF — confidently-wrong slices, worst first",
            "=" * 68,
            f"dataset:     {self.dataset_size} examples, {self.n_classes} classes",
            f"overall:     accuracy {self.overall.accuracy:.3f}, "
            f"mean confidence {self.overall.mean_confidence:.3f}, "
            f"gap {self.overall.gap:+.3f}, ECE {self.overall.ece:.3f}",
            f"cards:       {len(self.cards)}",
            "-" * 68,
        ]
        if not self.cards:
            header.append("no slice passed the configured thresholds")
        body = []
        for rank, card in enumerate(self.cards, start=1):
            body.append(f"#{rank}  score {card.score:.3f}  (gap x log size)")
            body.append(card.summary())
            body.append("-" * 68)
        return "\n".join(header + body)

    def __str__(self) -> str:
        return self.summary()

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]


def build_card(
    definition: tuple[Predicate, ...],
    indices: np.ndarray,
    stats: SliceStats,
    *,
    overall: SliceStats,
    dataset_size: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list | tuple | None,
) -> FailureCard:
    """Assemble a :class:`FailureCard` from a definition and its stats."""
    names = list(class_names) if class_names is not None else None

    def name_of(label: int) -> str:
        return str(names[label]) if names is not None else str(label)

    top_confusion: tuple[str, str, int] | None = None
    wrong = indices[y_true[indices] != y_pred[indices]]
    if wrong.size:
        pairs, counts = np.unique(
            np.stack([y_true[wrong], y_pred[wrong]]), axis=1, return_counts=True
        )
        best = int(np.argmax(counts))
        top_confusion = (
            name_of(int(pairs[0, best])),
            name_of(int(pairs[1, best])),
            int(counts[best]),
        )
    return FailureCard(
        definition=definition_text(definition),
        predicates=tuple(p.text for p in definition),
        indices=tuple(int(i) for i in indices),
        size=stats.size,
        dataset_size=dataset_size,
        errors=stats.errors,
        accuracy=stats.accuracy,
        mean_confidence=stats.mean_confidence,
        gap=stats.gap,
        ece=stats.ece,
        score=stats.score,
        overall_accuracy=overall.accuracy,
        overall_mean_confidence=overall.mean_confidence,
        top_confusion=top_confusion,
    )
