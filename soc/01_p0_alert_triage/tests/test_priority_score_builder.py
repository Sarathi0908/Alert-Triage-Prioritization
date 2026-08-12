"""Tests for src/features/priority_score_builder.py.

The contract that matters: priority ranks ALERTS ONLY, the weights are honoured,
and a builder restored from the model bundle reproduces the ranking the training
run produced — otherwise the queue order silently differs between offline
evaluation and live serving.
"""
import numpy as np
import pandas as pd

from src.features.priority_score_builder import PriorityScoreBuilder

WEIGHTS = {"probability": 0.70, "volume": 0.30}


def _builder(n_deciles=10):
    return PriorityScoreBuilder(WEIGHTS, "SRC_TO_DST_SECOND_BYTES", n_deciles, True)


def _fitted():
    b = _builder()
    probs = np.linspace(0.15, 0.99, 200)
    vol = np.log1p(np.linspace(0, 50_000, 200))
    return b.fit(probs, vol), probs, vol


def test_volume_is_log1p_of_the_named_feature():
    b = _builder()
    frame = pd.DataFrame({"SRC_TO_DST_SECOND_BYTES": [0, 999]})
    assert np.allclose(b.volume_of(frame), np.log1p([0, 999]))


def test_priority_is_bounded_0_to_100():
    b, probs, vol = _fitted()
    pri = b.transform_ranked(probs, vol)
    assert pri.min() >= 0.0 and pri.max() <= 100.0


def test_probability_dominates_the_blend():
    """0.70/0.30 means the higher-probability flow wins when volume is equal."""
    b, _, _ = _fitted()
    low = b.transform([0.20], [np.log1p(10_000)])[0]
    high = b.transform([0.99], [np.log1p(10_000)])[0]
    assert high > low


def test_volume_breaks_ties_at_equal_probability():
    b, _, _ = _fitted()
    quiet = b.transform([0.50], [np.log1p(1)])[0]
    loud = b.transform([0.50], [np.log1p(50_000)])[0]
    assert loud > quiet


def test_deciles_span_the_configured_range():
    b, probs, vol = _fitted()
    dec = b.decile(b.transform(probs, vol))
    assert dec.min() >= 1 and dec.max() <= b.n_deciles


def test_higher_priority_lands_in_a_higher_decile():
    b, _, _ = _fitted()
    lo = b.decile(b.transform([0.16], [np.log1p(10)]))[0]
    hi = b.decile(b.transform([0.99], [np.log1p(49_000)]))[0]
    assert hi >= lo


def test_bundle_round_trip_reproduces_the_ranking():
    """Serving restores the builder from the bundle; it must score identically."""
    b, probs, vol = _fitted()
    restored = PriorityScoreBuilder.from_bundle(b.to_bundle())
    assert np.allclose(b.transform(probs, vol), restored.transform(probs, vol))
    assert np.array_equal(b.decile(b.transform(probs, vol)),
                          restored.decile(restored.transform(probs, vol)))


def test_bundle_carries_the_grids_serving_needs():
    b, _, _ = _fitted()
    blob = b.to_bundle()
    for key in ("p_grid", "vol_grid", "rank_q", "decile_edges", "vol_feature",
                "priority_weights"):
        assert key in blob, f"bundle missing {key}"
    assert len(blob["p_grid"]) == len(blob["rank_q"]) == 101


def test_unfitted_builder_does_not_crash_on_transform():
    """Before fit there are no grids; percentiles must degrade to 0, not raise."""
    b = _builder()
    assert np.allclose(b.transform([0.9], [5.0]), [0.0])


def test_formula_names_the_volume_feature():
    b = _builder()
    f = b.formula()
    assert "SRC_TO_DST_SECOND_BYTES" in f and "0.7" in f
