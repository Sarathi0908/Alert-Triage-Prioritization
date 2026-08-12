"""Screen-invariant tests — the leak / noise / balance rules must hold on the
feature set the model actually trains on (data/native_selected.json).

These are the rules from configs/feature_config.yaml `screen`. They exist because
a silent screen regression (a leak sneaking back in, or one feature taking over
the gain) produces a model that looks better and is worse.
"""
import pytest

from src.config import DATA_DIR


def test_a_feature_set_exists(selected):
    assert selected["n"] == len(selected["features"]) > 0
    assert len(set(selected["features"])) == selected["n"], "duplicate feature in the set"


def test_no_leak_survived_the_screen(selected, cfg):
    """No kept feature may separate the classes almost perfectly on its own."""
    ceiling = cfg["feature"]["screen"]["leak"]["single_feature_auc_min"]
    offenders = {f: a for f, a in selected["single_feature_auc"].items() if a >= ceiling}
    assert not offenders, f"leak-grade features kept: {offenders}"


def test_hand_excluded_columns_are_absent(selected, cfg):
    excluded = set(cfg["feature"].get("exclude") or [])
    assert not (excluded & set(selected["features"])), (
        f"hand-excluded column present: {excluded & set(selected['features'])}")


def test_gain_is_balanced(selected, cfg):
    """No dominator above the cap, no dead weight below the floor."""
    bal = cfg["feature"]["screen"]["balance"]
    gains = selected["gain_pct"]
    assert gains, "no gain recorded for the selected features"
    assert max(gains.values()) <= bal["gain_cap_pct"], (
        f"feature dominates the model: {max(gains, key=gains.get)} "
        f"at {max(gains.values())}% > {bal['gain_cap_pct']}%")
    assert min(gains.values()) >= bal["gain_floor_pct"], (
        f"dead-weight feature kept: {min(gains, key=gains.get)} "
        f"at {min(gains.values())}% < {bal['gain_floor_pct']}%")


def test_minimum_feature_count_respected(selected, cfg):
    assert selected["n"] >= cfg["feature"]["screen"]["balance"]["min_selected"]


def test_every_feature_has_a_family(selected, cfg):
    """Family coverage is how we check a behavioural class has not lost its signal."""
    families, default = cfg["feature"]["families"], cfg["feature"]["default_family"]
    known = set(families) | {default}
    assert set(selected["features"]) == set(selected["families"])
    assert set(selected["families"].values()) <= known


def test_behavioural_families_are_diverse(selected):
    """A set drawn from a single family detects a single kind of attack."""
    assert len(set(selected["families"].values())) >= 2, (
        f"all features are one family: {selected['families']}")


def test_removal_reasons_are_recorded(selected):
    """Every dropped column must carry a reason — the audit is the deliverable."""
    removed = selected["removed"]
    assert removed, "no removal audit recorded"
    kept = set(selected["features"])
    for reason, cols in removed.items():
        assert not (set(cols) & kept), f"{reason} lists a KEPT feature: {set(cols) & kept}"


def test_no_categorical_features(selected):
    """This model is native-numeric only: the request body maps 1:1 to features."""
    assert selected["categorical"] == []


def test_screen_audit_csv_covers_the_selected_set(selected):
    """feature_screen.csv is the human-readable audit; it must not disagree."""
    audit = DATA_DIR / "feature_screen.csv"
    if not audit.exists():
        pytest.skip("feature_screen.csv missing — run src/features/feature_screener.py")
    import pandas as pd
    scr = pd.read_csv(audit)
    assert {"feature", "role", "gain_pct", "single_feature_auc"} <= set(scr.columns)
    marked = set(scr[scr.role == "SELECTED"].feature)
    assert marked == set(selected["features"]), (
        f"audit/selection disagree: only-in-audit={marked - set(selected['features'])}, "
        f"only-in-json={set(selected['features']) - marked}")
