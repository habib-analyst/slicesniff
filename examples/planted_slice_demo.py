"""Planted confidently-wrong slice demo (synthetic, mechanics only).

Builds a synthetic 3-class problem where the "model" is right ~90% of the
time at moderate confidence -- EXCEPT on one planted slice:

    bright images (brightness >= 0.75) from source "scanner_b"

where it is wrong 55% of the time while claiming ~0.92 confidence.  Then
SliceSniff has to find that slice from predictions + metadata alone and
rank it first.  (A second, honest hard slice -- dark images at genuinely
low confidence -- is included to show the gap ranking separating the two:
high error alone is not what SliceSniff hunts.)

This is a mechanics demo on synthetic data with a fixed seed, not a
benchmark.  The script prints whatever it measures and exits non-zero if
the planted slice is not the top card.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from slicesniff import mine_slices  # noqa: E402


def build_problem(seed: int = 0, n: int = 1200, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    # Bimodal brightness (a dim mode and a bright mode), so a quantile
    # band can actually isolate the bright population the slice lives in.
    bright_mode = rng.random(n) < 0.3
    brightness = np.where(
        bright_mode, rng.normal(0.86, 0.06, size=n), rng.normal(0.35, 0.13, size=n)
    )
    brightness = np.clip(brightness, 0.0, 1.0)
    sources = np.array(rng.choice(["lab", "scanner_a", "scanner_b"], size=n), dtype=object)

    planted = (brightness >= 0.75) & (sources == "scanner_b")
    dark = brightness < 0.2
    y_pred = y_true.copy()
    confidence = np.empty(n)
    for i in range(n):
        if planted[i]:
            correct, conf = rng.random() > 0.55, rng.uniform(0.88, 0.97)
        elif dark[i]:
            correct, conf = rng.random() > 0.35, rng.uniform(0.35, 0.55)
        else:
            correct, conf = rng.random() > 0.10, rng.uniform(0.70, 0.90)
        if not correct:
            others = [c for c in range(n_classes) if c != y_true[i]]
            y_pred[i] = others[rng.integers(0, len(others))]
        confidence[i] = conf

    probs = np.zeros((n, n_classes))
    for i in range(n):
        weights = rng.random(n_classes - 1) + 0.05
        rest = (1.0 - confidence[i]) * weights / weights.sum()
        others = [c for c in range(n_classes) if c != y_pred[i]]
        probs[i, y_pred[i]] = confidence[i]
        for c, value in zip(others, rest):
            probs[i, c] = value

    metadata = {"brightness": brightness, "source": sources}
    return probs, y_true, y_pred, metadata, planted


def main() -> int:
    probs, y_true, y_pred, metadata, planted = build_problem()
    report = mine_slices(probs, y_true, metadata)
    print(report.summary())

    top = report.cards[0] if report.cards else None
    found = (
        top is not None
        and "brightness" in top.definition
        and "scanner_b" in top.definition
    )
    if top is not None:
        top_set = set(top.indices)
        planted_set = set(int(i) for i in np.flatnonzero(planted))
        recall = len(top_set & planted_set) / max(len(planted_set), 1)
        precision = len(top_set & planted_set) / max(len(top_set), 1)
        print(
            f"planted slice recovery: recall {recall:.2f}, precision {precision:.2f} "
            f"({len(planted_set)} planted examples)"
        )
    else:
        recall = precision = 0.0

    out_dir = pathlib.Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["planted_slice"] = {
        "definition": "brightness >= 0.75 AND source == 'scanner_b'",
        "size": int(planted.sum()),
        "found_as_top_card": bool(found),
        "top_card_recall": recall,
        "top_card_precision": precision,
    }
    (out_dir / "demo_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"summary written to {out_dir / 'demo_summary.json'}")
    print("VERDICT:", "planted slice found as top card" if found else "PLANTED SLICE NOT FOUND")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
