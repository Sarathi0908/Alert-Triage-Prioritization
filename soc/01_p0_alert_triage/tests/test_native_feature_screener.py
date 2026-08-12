"""Tests for src/features/native_feature_screener.py.

Two halves: the screen's rules run against toy data where the right answer is
known, and the committed feature set (data/native_selected.json) is checked to
still satisfy those rules. A silent screen regression — a leak sneaking back in,
or one feature taking over the gain — produces a model that looks better and is
worse, so these are invariants, not smoke tests.
"""
import pytest

from src.config import DATA_DIR
from src.features import native_feature_screener as fs


# ---- the rules, on toy data ---------------------------------------------------

def test_family_of_matches_in_config_order():
    families = {"rate": ["throughput", "second_bytes"], "volume": ["byte"]}
    # SRC_TO_DST_SECOND_BYTES matches both; "rate" is declared first and must win.
    assert fs.family_of("SRC_TO_DST_SECOND_BYTES", families, "other") == "rate"
    assert fs.family_of("IN_BYTES", families, "other") == "volume"
    assert fs.family_of("MIN_TTL", families, "other") == "other"


def test_profile_reports_zero_fraction_and_solo_auc(toy_flows, cfg):
    y = toy_flows["y_is_attack"].values
    cols = ["L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES", "DEAD_COLUMN"]
    prof = fs.profile_features(toy_flows, y, cols, cfg["feature"]["families"], "other")
    # Bracket access, not attribute: `row.unique` would resolve to Series.unique.
    dead = prof.set_index("feature").loc["DEAD_COLUMN"]
    assert dead["zero_frac"] == 1.0 and dead["unique"] == 1
    strong = prof.set_index("feature").loc["SRC_TO_DST_SECOND_BYTES"]
    assert strong["single_feature_auc"] > 0.9      # separates by construction


def test_screen_drops_the_near_constant_column(toy_flows, cfg):
    y = toy_flows["y_is_attack"].values
    cols = ["L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES", "DEAD_COLUMN"]
    selected, audit, payload = fs.screen(toy_flows, y, cols, cfg["feature"],
                                        seed=0, log=lambda *_: None)
    assert "DEAD_COLUMN" not in selected
    assert "DEAD_COLUMN" in payload["removed"]["near_constant_noise"]
    assert set(audit.role) <= {"SELECTED", "near-constant NOISE", "LEAK",
                               "redundant NOISE", "DOMINANT (removed for balance)",
                               "low-gain tail", "not-selected"}


def test_screen_payload_is_self_consistent(toy_flows, cfg):
    y = toy_flows["y_is_attack"].values
    cols = ["L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES", "DEAD_COLUMN"]
    selected, _, payload = fs.screen(toy_flows, y, cols, cfg["feature"],
                                     seed=0, log=lambda *_: None)
    assert payload["n"] == len(payload["features"]) == len(selected)
    assert set(payload["gain_pct"]) == set(selected)
    assert set(payload["families"]) == set(selected)
    for reason, dropped in payload["removed"].items():
        assert not (set(dropped) & set(selected)), f"{reason} lists a kept feature"


# ---- the committed feature set still obeys the rules -------------------------

def test_feature_set_exists_and_has_no_duplicates(selected):
    assert selected["n"] == len(selected["features"]) > 0
    assert len(set(selected["features"])) == selected["n"]


def test_no_leak_survived_the_screen(selected, cfg):
    ceiling = cfg["feature"]["screen"]["leak"]["single_feature_auc_min"]
    offenders = {f: a for f, a in selected["single_feature_auc"].items() if a >= ceiling}
    assert not offenders, f"leak-grade features kept: {offenders}"


def test_hand_excluded_columns_are_absent(selected, cfg):
    excluded = set(cfg["feature"].get("exclude") or [])
    assert not (excluded & set(selected["features"]))


def test_gain_is_balanced(selected, cfg):
    """No dominator above the cap, no dead weight below the floor."""
    bal = cfg["feature"]["screen"]["balance"]
    gains = selected["gain_pct"]
    assert max(gains.values()) <= bal["gain_cap_pct"], (
        f"{max(gains, key=gains.get)} dominates at {max(gains.values())}%")
    assert min(gains.values()) >= bal["gain_floor_pct"], (
        f"{min(gains, key=gains.get)} is dead weight at {min(gains.values())}%")


def test_behavioural_families_are_diverse(selected):
    """A set drawn from one family detects one kind of attack."""
    assert len(set(selected["families"].values())) >= 2, selected["families"]


def test_no_categorical_features(selected):
    """Native-numeric only: the request body maps 1:1 onto the features."""
    assert selected["categorical"] == []


def test_audit_csv_agrees_with_the_selection(selected):
    audit = DATA_DIR / "feature_screen.csv"
    if not audit.exists():
        pytest.skip("feature_screen.csv missing — run src/pipeline.py")
    import pandas as pd
    scr = pd.read_csv(audit)
    assert {"feature", "role", "gain_pct", "single_feature_auc"} <= set(scr.columns)
    marked = set(scr[scr.role == "SELECTED"].feature)
    assert marked == set(selected["features"])
