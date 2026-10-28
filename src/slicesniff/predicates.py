"""Human-readable slice predicates over metadata columns.

A predicate selects a boolean mask over the dataset.  Three kinds exist:

- **categorical** -- ``feature == value`` or ``feature in {v1, v2, ...}``
- **numeric band** -- ``lo <= feature <= hi`` where the edges come from
  quantiles of the observed values, so the text names real numbers
- **numeric split** -- ``feature <= t`` or ``feature > t`` at a quantile

A *slice definition* is a tuple of predicates, read as a conjunction.
Definitions have a canonical (sorted) order so equality and hashing are
stable, which is what makes failure cards reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Predicate", "generate_predicates", "apply_definition"]


def _format_number(value: float) -> str:
    return f"{value:.4g}"


def _format_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_number(float(value))
    return f"'{value}'"


@dataclass(frozen=True, eq=False)
class Predicate:
    """One readable condition on one metadata column.

    Identity equality/hash: predicates carry an ndarray mask (unhashable),
    and the miner always reuses the same generated objects, so identity
    is the correct -- and stable -- notion of sameness for definitions.
    """

    feature: str
    text: str
    mask: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.text


def _is_categorical(values: np.ndarray) -> bool:
    return values.dtype.kind in "OSU" or values.dtype == bool


def _value_counts(values: np.ndarray) -> list[tuple[object, int]]:
    uniques, counts = np.unique(values, return_counts=True)
    return sorted(zip(uniques.tolist(), (int(c) for c in counts)), key=lambda kv: -kv[1])


def generate_predicates(
    feature: str,
    values: np.ndarray,
    *,
    n: int,
    min_support: int,
    max_categories: int = 12,
    n_bins: int = 4,
    include_missing: bool = True,
) -> list[Predicate]:
    """Generate every level-1 predicate for one metadata column.

    Predicates whose support is below ``min_support`` or covers (nearly)
    the whole dataset are dropped: they cannot define an interesting slice.
    """
    values = np.asarray(values)
    if values.shape[0] != n:
        raise ValueError(f"metadata column {feature!r} has length {values.shape[0]}, expected {n}")
    predicates: list[Predicate] = []

    def keep(mask: np.ndarray) -> bool:
        support = int(mask.sum())
        return min_support <= support <= n - min_support

    if _is_categorical(values):
        counts = _value_counts(values)
        if len(counts) > max_categories:
            counts = counts[:max_categories]
        for value, support in counts:
            if support < min_support:
                continue
            mask = values == value
            if keep(mask):
                predicates.append(
                    Predicate(feature, f"{feature} == {_format_value(value)}", mask)
                )
            complement = ~mask
            if support <= n / 2 and keep(complement):
                predicates.append(
                    Predicate(feature, f"{feature} != {_format_value(value)}", complement)
                )
        return predicates

    numeric = values.astype(float)
    finite = np.isfinite(numeric)
    if include_missing and not bool(finite.all()):
        mask = ~finite
        if keep(mask):
            predicates.append(Predicate(feature, f"{feature} is missing", mask))
    observed = numeric[finite]
    if observed.size < max(2 * min_support, 2):
        return predicates
    quantiles = np.unique(np.quantile(observed, np.linspace(0.0, 1.0, n_bins + 1)))
    if quantiles.size < 2:
        return predicates
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        mask = finite & (numeric >= lo) & (numeric <= hi)
        if lo == hi:
            text = f"{feature} == {_format_number(float(lo))}"
        else:
            text = f"{_format_number(float(lo))} <= {feature} <= {_format_number(float(hi))}"
        if keep(mask):
            predicates.append(Predicate(feature, text, mask))
    for threshold in np.unique(np.quantile(observed, [0.25, 0.5, 0.75])):
        for mask, text in (
            (finite & (numeric <= threshold), f"{feature} <= {_format_number(float(threshold))}"),
            (finite & (numeric > threshold), f"{feature} > {_format_number(float(threshold))}"),
        ):
            if keep(mask):
                predicates.append(Predicate(feature, text, mask))
    return predicates


def apply_definition(definition: tuple[Predicate, ...], n: int) -> np.ndarray:
    """Boolean mask of the conjunction of ``definition`` over ``n`` rows."""
    mask = np.ones(n, dtype=bool)
    for predicate in definition:
        mask &= predicate.mask
    return mask


def definition_text(definition: tuple[Predicate, ...]) -> str:
    """Canonical readable text for a slice definition."""
    return " AND ".join(p.text for p in definition)
