"""Tests for SliceSniff.

The centrepiece is the planted-slice test: a synthetic problem with one
deliberately planted confidently-wrong slice, which the miner MUST find
and rank first, plus an honest hard slice (high error, genuinely low
confidence) which must NOT be mistaken for overconfidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from planted_slice_demo import build_problem  # noqa: E402

from slicesniff import image_stats, mine_slices, prediction_metadata  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    return build_problem()


@pytest.fixture(scope="module")
def report(problem):
    probs, y_true, _, metadata, _ = problem
    return mine_slices(probs, y_true, metadata)


def test_planted_slice_is_top_card(problem, report):
    _, _, _, _, planted = problem
    top = report.cards[0]
    assert "brightness" in top.definition
    assert "scanner_b" in top.definition
    planted_idx = set(int(i) for i in np.flatnonzero(planted))
    top_idx = set(top.indices)
    recall = len(top_idx & planted_idx) / len(planted_idx)
    precision = len(top_idx & planted_idx) / len(top_idx)
    assert recall >= 0.75
    assert precision >= 0.9


def test_top_card_is_confidently_wrong(report):
    top = report.cards[0]
    assert top.accuracy < 0.60
    assert top.mean_confidence > 0.85
    assert top.gap > 0.35
    assert top.errors > top.size // 3


def test_honest_hard_slice_is_not_flagged(report):
    # The planted *dark* slice has high error but low confidence by
    # construction; no card may pair low accuracy with low confidence.
    for card in report.cards:
        assert not (card.accuracy < 0.6 and card.mean_confidence < 0.6)


def test_cards_sorted_and_reproducible(problem):
    probs, y_true, _, metadata, _ = problem
    first = mine_slices(probs, y_true, metadata)
    second = mine_slices(probs, y_true, metadata)
    scores = [card.score for card in first.cards]
    assert scores == sorted(scores, reverse=True)
    assert first.to_dict() == second.to_dict()


def test_card_indices_match_definition(problem, report):
    probs, y_true, _, metadata, _ = problem
    y_pred = np.argmax(probs, axis=1)
    top = report.cards[0]
    idx = np.array(top.indices)
    recomputed = float((y_pred[idx] == y_true[idx]).mean())
    assert recomputed == pytest.approx(top.accuracy)
    assert len(set(top.indices)) == top.size
    assert top.coverage == pytest.approx(top.size / len(y_true))


def test_json_round_trip(report):
    payload = json.loads(report.to_json())
    assert payload["dataset_size"] == report.dataset_size
    assert payload["cards"][0]["definition"] == report.cards[0].definition
    assert payload["cards"][0]["indices"] == list(report.cards[0].indices)
    assert payload["config"]["features_searched"]


def test_overall_stats_sane(report):
    assert 0.78 < report.overall.accuracy < 0.88
    assert report.overall.ece > 0.05
    assert report.dataset_size == 1200
    assert report.n_classes == 3


def _tiny_problem(seed=1, n=600):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    group = np.array(rng.choice(["a", "b"], size=n), dtype=object)
    y_pred = y_true.copy()
    conf = rng.uniform(0.7, 0.9, size=n)
    flip = (group == "b") & (rng.random(n) < 0.5)
    y_pred[flip] = 1 - y_true[flip]
    conf = np.where(flip, rng.uniform(0.85, 0.95, n), conf)
    probs = np.zeros((n, 2))
    probs[np.arange(n), y_pred] = conf
    probs[np.arange(n), 1 - y_pred] = 1.0 - conf
    return probs, y_true, {"group": group}


def test_ypred_confidence_path():
    probs, y_true, metadata = _tiny_problem()
    y_pred = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(y_true)), y_pred]
    via_probs = mine_slices(probs, y_true, metadata)
    via_hard = mine_slices(y_true=y_true, metadata=metadata, y_pred=y_pred, confidence=confidence)
    assert via_hard.cards
    assert via_hard.cards[0].definition == via_probs.cards[0].definition


def test_confident_model_yields_no_cards():
    rng = np.random.default_rng(3)
    n = 400
    y_true = rng.integers(0, 3, size=n)
    probs = np.full((n, 3), 0.05)
    probs[np.arange(n), y_true] = 0.9
    metadata = {"shade": np.array(rng.choice(["x", "y"], size=n), dtype=object)}
    result = mine_slices(probs, y_true, metadata)
    assert result.cards == []
    assert result.overall.accuracy == 1.0


def test_string_labels_and_class_names():
    probs, y_true, metadata = _tiny_problem()
    names = ["cat", "dog"]
    str_true = np.array([names[int(k)] for k in y_true], dtype=object)
    result = mine_slices(probs, str_true, metadata, class_names=names)
    assert result.cards
    assert result.cards[0].top_confusion is not None
    assert result.cards[0].top_confusion[0] in names


def test_min_gap_filters_everything(problem):
    probs, y_true, _, metadata, _ = problem
    result = mine_slices(probs, y_true, metadata, min_gap=0.99)
    assert result.cards == []


def test_no_metadata_no_prediction_features_is_empty(problem):
    probs, y_true, _, _, _ = problem
    result = mine_slices(probs, y_true, None)
    assert result.cards == []
    assert result.config["features_searched"] == []


def test_prediction_features_opt_in(problem):
    probs, y_true, _, metadata, _ = problem
    result = mine_slices(probs, y_true, metadata, include_prediction_features=True)
    assert "confidence_band" in result.config["features_searched"]
    assert "margin" in result.config["features_searched"]


def test_input_validation(problem):
    probs, y_true, _, metadata, _ = problem
    with pytest.raises(ValueError):
        mine_slices(probs * 2, y_true, metadata)  # rows no longer sum to 1
    with pytest.raises(ValueError):
        mine_slices(probs, y_true[:-1], metadata)  # length mismatch
    with pytest.raises(ValueError):
        mine_slices(probs, y_true, {"bad": np.zeros(5)})  # metadata length
    with pytest.raises(ValueError):
        mine_slices(probs, y_true, metadata, y_pred=np.argmax(probs, axis=1),
                    confidence=probs.max(axis=1))  # both input styles
    with pytest.raises(ValueError):
        mine_slices(y_true=y_true, metadata=metadata, y_pred=np.argmax(probs, axis=1),
                    confidence=np.full(len(y_true), 1.5))  # confidence out of range
    bad = probs.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        mine_slices(bad, y_true, metadata)
    with pytest.raises(ValueError):
        mine_slices(probs, y_true, metadata, min_support=1)


def test_sniff_model_with_stub():
    class Stub:
        def __init__(self, probs):
            self._probs = probs

        def predict_proba(self, X):
            return self._probs

    probs, y_true, metadata = _tiny_problem()
    from slicesniff import sniff_model

    report = sniff_model(Stub(probs), None, y_true, metadata)
    assert report.cards
    assert "group == 'b'" in report.cards[0].definition
    with pytest.raises(ValueError):
        sniff_model(object(), None, y_true, metadata)


def test_image_stats_shapes_and_ranges():
    rng = np.random.default_rng(0)
    gray = rng.random((12, 16, 16))
    stats = image_stats(gray)
    assert set(stats) == {
        "brightness", "contrast", "edge_density", "border_center_ratio", "height", "width",
    }
    assert all(values.shape == (12,) for values in stats.values())
    assert np.allclose(stats["height"], 16)
    color = rng.random((5, 10, 8, 3))
    stats_color = image_stats(color)
    assert stats_color["brightness"].shape == (5,)
    assert (stats_color["border_center_ratio"] >= 0).all()
    with pytest.raises(ValueError):
        image_stats(np.zeros((4, 4)))


def test_prediction_metadata_bands():
    y_pred = np.array([0, 1, 1, 0])
    confidence = np.array([0.4, 0.6, 0.8, 0.95])
    meta = prediction_metadata(y_pred, confidence, top2_margin=np.array([0.1, 0.2, 0.3, 0.4]))
    assert list(meta["confidence_band"]) == ["low", "mid", "high", "very_high"]
    assert list(meta["predicted_class"]) == [0, 1, 1, 0]
    named = prediction_metadata(y_pred, confidence, class_names=["a", "b"])
    assert list(named["predicted_class"]) == ["a", "b", "b", "a"]
