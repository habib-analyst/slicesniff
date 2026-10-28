# SliceSniff

**Sniff out the slices where your classifier is confidently wrong — one call on its predictions returns ranked, reproducible failure cards: the slice definition, how wrong it is, how confident it stayed, and the exact example indices.**

A model can post 90% accuracy and still be a liability on one coherent corner of the data — dim images from one scanner, one source, one size band — where it is wrong half the time *while reporting 0.92 confidence*. Aggregate metrics hide that corner; per-example scores drown you in rows. SliceSniff mines it directly: give it the model's predicted probabilities, the true labels, and whatever metadata you have (source tags, classes, image statistics), and it searches conjunctions of simple human-readable predicates — `0.80 <= brightness <= 1 AND source == 'scanner_b'` — ranking slices by the **overconfidence gap** (mean confidence − accuracy), and emits one **failure card** per bad slice with everything needed to reproduce it.

## Install

SliceSniff is not on PyPI yet. Install the GitHub version:

```bash
pip install "git+https://github.com/habib-analyst/slicesniff.git"
```

For development:

```bash
git clone https://github.com/habib-analyst/slicesniff.git
cd slicesniff
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.10+. The only runtime dependency is NumPy.

## Quickstart

```python
import numpy as np
from slicesniff import mine_slices

report = mine_slices(probs, y_true, {"source": sources, "brightness": brightness})
print(report.summary())

card = report.cards[0]
print(card.definition)        # "0.8017 <= brightness <= 1 AND source == 'scanner_b'"
print(card.accuracy, card.mean_confidence, card.gap)
rows = X[card.indices]        # the exact failing examples, in your row order
```

Real output from `examples/planted_slice_demo.py`, which plants exactly this failure in synthetic data (overall accuracy 0.830) and makes SliceSniff find it:

```
#1  score 1.845  (gap x log size)
slice:       0.8017 <= brightness <= 1 AND source == 'scanner_b'
size:        114 of 1200 examples (9.5% of the data, indices in 'indices')
accuracy:    0.535  (overall 0.830)
confidence:  0.924  (overall 0.787)
gap:         +0.389  (confidence - accuracy; positive = overconfident)
errors:      53 of 114 wrong, in-slice ECE 0.389
top error:   true '1' predicted as '0' x14
```

Prefer hard predictions? Pass them instead of probabilities:

```python
report = mine_slices(y_true=y_true, metadata=metadata,
                     y_pred=y_pred, confidence=confidence)
```

Have a fitted scikit-learn-style model? `sniff_model(model, X, y_true, metadata)` calls its `predict_proba` for you. SliceSniff itself never imports your framework.

## How it works

- **Predicates.** Categorical columns generate `feature == value` / `!= value`; numeric columns generate quantile bands (`0.43 <= brightness <= 0.80`) and quantile splits (`brightness > 0.43`), so edges in card text are real numbers from your data. NaNs get an explicit `feature is missing` predicate.
- **Search.** All single predicates are scored, then a beam search refines the best ones into conjunctions (default depth 2, one predicate per feature). A slice needs at least `min_support` examples (default 30, and at least 1% of the data).
- **Ranking.** A slice becomes a card only if its overconfidence gap (mean confidence − accuracy) reaches `min_gap` (default 0.15). Cards are ranked by `gap × log1p(size)`, and near-duplicate cards (Jaccard overlap > 0.9) are suppressed, so the list reads worst-first without repeating itself.
- **The label is never a slice feature.** Definitions like `label == 0 AND predicted == 2` are wrong by construction; SliceSniff excludes `y_true` from the search space and instead reports each card's dominant confusion (`top error: true '1' predicted as '0' x14`).

Image statistics make good metadata. `slicesniff.image_stats(images)` turns an `(N, H, W)` or `(N, H, W, C)` batch into `brightness`, `contrast`, `edge_density`, `border_center_ratio`, `height`, `width` columns ready to drop into the metadata dict. Columns derived from the model's own output (`predicted_class`, `confidence_band`, `margin`) can be searched too, via `include_prediction_features=True` — off by default, because "the model fails when it is very confident" is true of almost every bad slice and would outrank the metadata slices that say *where* it fails.

## The failure card

Each `FailureCard` carries: `definition` and `predicates` (readable text), `indices` (exact example indices), `size` / `coverage`, `errors`, `accuracy`, `mean_confidence`, `gap`, in-slice `ece`, the ranking `score`, the overall-dataset baselines it was compared against, the dominant `top_confusion`, and — on the report — the full search `config`, so a card can be re-derived or challenged. `report.to_json()` serializes everything for CI gates, tickets, and regression tracking: re-run SliceSniff after a fix and diff the cards.

## What SliceSniff is not

- **Not an embedding-based slice discovery tool.** [Domino](https://pypi.org/project/domino/) clusters cross-modal embeddings and describes the clusters in natural language — the tool to reach for when the failing concept has *no* metadata column. SliceSniff deliberately searches only named columns, so every slice it returns is a checkable predicate, not a cluster you have to interpret.
- **Not a general slice finder.** [SliceLine](https://pypi.org/project/sliceline/) finds high-*error* slices over categorical features from a binary error vector; probabilities never enter it. SliceSniff's whole ranking is the accuracy-vs-confidence gap: a slice the model gets wrong *while doubting itself* is honest and is not what this hunts. (Predicate slice-finding as a technique is SliceLine's lineage and is credited as such.)
- **Not a dashboard.** Microsoft's Error Analysis (Responsible AI) offers an interactive error tree/heatmap over error rate. SliceSniff is the headless counterpart: one function call, a ranked list, a JSON artifact — built for pipelines and regression checks, not clicking.
- **Not a label-error detector.** [cleanlab](https://docs.cleanlab.ai/) scores *individual* examples for likely label issues via confident learning. SliceSniff works one level up — subgroups, not rows — and says nothing about which single label is wrong.
- **Not a fairness audit.** Fairlearn/AIF360 evaluate groups *you* define over protected attributes. SliceSniff mines groups you did *not* define; if a protected attribute is in your metadata it will be treated like any other column, and what you do with that card is a policy decision, not a metric.
- **Not a completeness guarantee.** The beam search is a heuristic over the columns you provide. A failure mode that no metadata column (or combination within `max_depth`) expresses will not be found, and SliceSniff will not tell you it exists.

## Measured demo (synthetic, reproducible)

`examples/planted_slice_demo.py` builds a 1,200-example, 3-class problem (seed 0) whose model is ~90% accurate at moderate confidence, except on one planted slice — bright images from `scanner_b`, 55% wrong at ~0.92 confidence — plus one *honest* hard slice (dark images, 35% wrong at genuinely low confidence). SliceSniff ranks the planted slice #1 (accuracy 0.535, confidence 0.924, gap +0.389), recovers 83% of the planted examples (the card's band edge is data-driven, 0.8017 vs the planted 0.75) at 100% precision, and correctly does not flag the dark slice. The script prints whatever it measures and exits non-zero if the planted slice is not the top card. It is a mechanics demo, not a benchmark.

## License

MIT. Issues and pull requests are welcome on GitHub.
