# Changelog

## 0.1.0 — 2026-10-28

First release.

- `mine_slices`: prediction-only slice mining over metadata columns.
  Accepts class probabilities, or hard predictions + confidences, plus a
  dict of categorical/numeric metadata; returns a `SniffReport` of ranked
  `FailureCard`s (definition, exact indices, accuracy, mean confidence,
  overconfidence gap, in-slice ECE, dominant confusion, search config).
- Beam search over human-readable predicates (categorical equality,
  quantile bands and splits, explicit missing-value predicates), ranked
  by `gap * log1p(size)` with Jaccard near-duplicate suppression.
- `sniff_model`: convenience wrapper for fitted models exposing
  `predict_proba`.
- `image_stats`: per-image brightness/contrast/edge-density/
  border-center-ratio/size columns from raw image batches.
- `prediction_metadata`: opt-in slicing over the model's own output
  (predicted class, confidence band, top-2 margin).
- The true label is never used as a slice feature (prevents tautological
  label × prediction slices).
- Synthetic planted-slice example that verifies the miner recovers a
  deliberately planted confidently-wrong slice as the top card.
