"""Tests for src/evaluation/flow_traffic_profiler.py.

The profiler's job is to make the screen auditable, so its counts must be right and
it must not overflow on the float32 byte counters that made the original EDA emit
RuntimeWarnings.
"""
import numpy as np
import pandas as pd

from src.evaluation import flow_traffic_profiler as ftp


def _frame():
    return pd.DataFrame({
        "a": [1.0, 2.0, 2.0, np.nan],
        "attack_class": ["BENIGN", "DoS", "DoS", "Bot"],
        "y_is_attack": [0, 1, 1, 1],
    })


def test_profile_counts_nulls_uniques_and_duplicates():
    p = ftp.profile(_frame(), ["a"]).set_index("column").loc["a"]
    assert p.null_count == 1
    assert p.unique_count == 2          # 1.0 and 2.0
    assert p.duplicate_count == 1       # only 2.0 appears more than once
    assert p.max_repeat == 2


def test_profile_reports_range_for_numeric_columns():
    p = ftp.profile(_frame(), ["a"]).set_index("column").loc["a"]
    assert p["min"] == 1.0 and p["max"] == 2.0


def test_profile_leaves_range_blank_for_non_numeric():
    p = ftp.profile(_frame(), ["attack_class"]).set_index("column").loc["attack_class"]
    assert p["min"] == "" and p["mean"] == ""


def test_profile_does_not_overflow_float32_counters():
    """Large byte counters stored as float32 overflow when summed in float32; the
    profiler must still report a finite mean."""
    big = pd.DataFrame({"bytes": np.full(100_000, 3.0e38, dtype=np.float32)})
    mean = ftp.profile(big, ["bytes"]).set_index("column").loc["bytes"]["mean"]
    assert np.isfinite(mean)


def test_attack_class_counts_excludes_benign():
    vc = ftp.attack_class_counts(_frame(), "attack_class")
    assert "BENIGN" not in vc.index
    assert vc["DoS"] == 2 and vc["Bot"] == 1


def test_single_feature_auc_is_direction_agnostic():
    """A perfectly inverted feature is as informative as a perfectly aligned one."""
    df = pd.DataFrame({"up": [1, 2, 3, 4], "down": [4, 3, 2, 1],
                       "y_is_attack": [0, 0, 1, 1]})
    auc = ftp.single_feature_auc(df, ["up", "down"], "y_is_attack")
    assert auc["up"] == 1.0 and auc["down"] == 1.0


def test_class_signature_is_z_scored_and_clipped():
    df = pd.DataFrame({"f": [1.0, 1.0, 100.0, 100.0],
                       "attack_class": ["BENIGN", "BENIGN", "DoS", "DoS"],
                       "y_is_attack": [0, 0, 1, 1]})
    vc = ftp.attack_class_counts(df, "attack_class")
    z = ftp.class_signature(df, ["f"], "attack_class", vc, clip=3.0)
    assert set(z.index) == {"BENIGN", "DoS"}
    assert z.values.min() >= -3.0 and z.values.max() <= 3.0
    assert z.loc["DoS", "f"] > z.loc["BENIGN", "f"]
