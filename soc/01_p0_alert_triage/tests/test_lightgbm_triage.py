"""Tests for src/models/lightgbm_triage.py.

The class contract: fit derives scale_pos_weight from the split (it is a property
of the data, not a knob), the threshold is fitted to a precision floor rather than
left at 0.5, explanations come from the booster itself, and save/load round-trips
everything serving depends on.
"""
import numpy as np
import pytest

from src.models.lightgbm_triage import LightGBMTriage

FEATURES = ["L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES"]
PARAMS = {"objective": "binary", "n_estimators": 40, "num_leaves": 8,
          "learning_rate": 0.2, "n_jobs": 1, "verbose": -1}


@pytest.fixture
def fitted(toy_flows):
    y = toy_flows["y_is_attack"].values
    return LightGBMTriage(PARAMS, seed=0).fit(toy_flows[FEATURES], FEATURES, y), toy_flows, y


def test_fit_derives_scale_pos_weight_from_the_split(fitted):
    model, _, y = fitted
    expected = (y == 0).sum() / (y == 1).sum()
    assert model.scale_pos_weight == pytest.approx(expected)


def test_probability_is_a_probability(fitted):
    model, flows, _ = fitted
    p = model.attack_probability(flows[FEATURES])
    assert p.shape == (len(flows),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_model_separates_the_toy_classes(fitted):
    model, flows, y = fitted
    p = model.attack_probability(flows[FEATURES])
    assert p[y == 1].mean() > p[y == 0].mean()


def test_fit_threshold_holds_the_precision_floor(fitted):
    model, flows, y = fitted
    p = model.attack_probability(flows[FEATURES])
    thr = model.fit_threshold(y, p, min_precision=0.90)
    from sklearn.metrics import precision_score
    assert precision_score(y, (p >= thr).astype(int), zero_division=0) >= 0.90


def test_fit_threshold_falls_back_when_the_floor_is_unreachable(fitted):
    """An impossible floor must return the stated fallback, not raise or silently
    pick something arbitrary."""
    model, flows, y = fitted
    p = model.attack_probability(flows[FEATURES])
    thr = model.fit_threshold(y, p, min_precision=1.0000001, fallback=0.5)
    assert thr == 0.5


def test_verdict_uses_the_fitted_threshold(fitted):
    model, flows, y = fitted
    p = model.attack_probability(flows[FEATURES])
    model.fit_threshold(y, p, min_precision=0.90)
    assert np.array_equal(model.verdict(p), (p >= model.threshold).astype(int))


def test_shap_contributions_shape_and_direction(fitted):
    """One column per feature, bias dropped; contributions push the score they should."""
    model, flows, y = fitted
    sv = model.shap_contributions(flows[FEATURES])
    assert sv.shape == (len(flows), len(FEATURES))
    # attack rows should carry more positive total contribution than benign rows
    assert sv[y == 1].sum(axis=1).mean() > sv[y == 0].sum(axis=1).mean()


def test_gain_importance_sums_to_100(fitted):
    model, _, _ = fitted
    gain = model.gain_importance()
    assert set(gain.index) == set(FEATURES)
    assert gain.sum() == pytest.approx(100.0)


def test_save_load_round_trip(fitted, tmp_path):
    model, flows, y = fitted
    p = model.attack_probability(flows[FEATURES])
    model.fit_threshold(y, p, min_precision=0.90)
    path = tmp_path / "m.pkl"
    model.save(str(path), extra={"vol_feature": "SRC_TO_DST_SECOND_BYTES"})
    restored, blob = LightGBMTriage.load(str(path))
    assert restored.feature_names == FEATURES
    assert restored.threshold == model.threshold
    assert blob["vol_feature"] == "SRC_TO_DST_SECOND_BYTES"
    assert np.allclose(restored.attack_probability(flows[FEATURES]), p)


def test_accepts_a_bare_array_in_feature_order(fitted):
    """Serving hands over a DataFrame, but a raw array in feature order must work."""
    model, flows, _ = fitted
    from_frame = model.attack_probability(flows[FEATURES])
    from_array = model.attack_probability(flows[FEATURES].values)
    assert np.allclose(from_frame, from_array)


def test_trained_bundle_on_disk_matches_its_metrics(trained, metrics):
    """The shipped bundle's threshold is the one the run recorded."""
    model, blob = trained
    assert round(model.threshold, 6) == metrics["threshold"]
    assert model.feature_names == metrics["features"]
    assert blob["priority_weights"]["probability"] + blob["priority_weights"]["volume"] == 1.0
