"""Tests for src/evaluation/operating_point_validator.py.

Each gate must fail when it should. A validator that only ever passes is worse
than no validator, so every gate is exercised in both directions.
"""
import math

import pytest

from src.evaluation import operating_point_validator as opv

MODEL_CFG = {
    "operating_point": {"min_precision": 0.70},
    "acceptance": {"roc_auc_min": 0.95, "pr_auc_min": 0.90, "fn_rate_max": 0.05,
                   "precision_min": 0.70, "roc_gap_max": 0.05,
                   "alert_reduction_min": 0.50, "fp_per_analyst_hour_max": 4.0,
                   "min_recall_any_class": 0.50},
}
SERVING_CFG = {"scoring": {"analyst_minutes_per_alert": 5}}


def _good(**over):
    m = {"roc_auc": 0.99, "pr_auc": 0.98, "precision": 0.75, "fn_rate_%": 1.5,
         "roc_gap": 0.002, "alert_reduction_%": 68.0, "tp": 900, "fp": 100}
    m.update(over)
    return m


def _classes(min_recall=0.9):
    return [{"attack_class": "DoS", "n": 100, "recall": 0.99},
            {"attack_class": "Bot", "n": 10, "recall": min_recall}]


def test_good_metrics_pass_every_gate():
    g = opv.evaluate_gates(_good(), MODEL_CFG, SERVING_CFG, _classes())
    assert g["all_pass"]


def test_low_roc_fails():
    g = opv.evaluate_gates(_good(roc_auc=0.80), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["roc_pass"] and not g["all_pass"]


def test_low_pr_auc_fails():
    g = opv.evaluate_gates(_good(pr_auc=0.50), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["pr_pass"] and not g["all_pass"]


def test_high_fn_rate_fails():
    g = opv.evaluate_gates(_good(**{"fn_rate_%": 30.0}), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["fn_pass"] and not g["all_pass"]


def test_precision_below_the_fitted_floor_fails():
    g = opv.evaluate_gates(_good(precision=0.40), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["precision_pass"] and not g["all_pass"]


def test_wide_train_test_gap_fails_as_overfit():
    g = opv.evaluate_gates(_good(roc_gap=0.30), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["not_overfit_pass"] and not g["all_pass"]


def test_no_alert_reduction_fails():
    g = opv.evaluate_gates(_good(**{"alert_reduction_%": 5.0}), MODEL_CFG,
                           SERVING_CFG, _classes())
    assert not g["reduction_pass"] and not g["all_pass"]


def test_a_single_blind_spot_class_fails_coverage():
    """Aggregate is fine, one class is never caught — must not pass."""
    g = opv.evaluate_gates(_good(), MODEL_CFG, SERVING_CFG, _classes(min_recall=0.05))
    assert not g["class_coverage_pass"] and not g["all_pass"]
    assert g["weakest_attack_class"] == "Bot"


def test_fp_per_analyst_hour_arithmetic():
    """1000 queued alerts at 5 min each = 83.33 analyst-hours; 100 FP -> 1.2/hr."""
    fp_hr = opv.fp_per_analyst_hour({"tp": 900, "fp": 100}, SERVING_CFG)
    assert fp_hr == pytest.approx(100 / (1000 * 5 / 60.0))


def test_fp_per_analyst_hour_of_an_empty_queue_is_nan_not_a_crash():
    assert math.isnan(opv.fp_per_analyst_hour({"tp": 0, "fp": 0}, SERVING_CFG))


def test_workload_ceiling_fails_when_analysts_are_swamped():
    g = opv.evaluate_gates(_good(tp=1, fp=999), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["fp_pass"] and not g["all_pass"]


def test_missing_optional_gates_do_not_block():
    """A config without the optional ceilings must still be evaluable."""
    cfg = {"operating_point": {"min_precision": 0.70},
           "acceptance": {k: v for k, v in MODEL_CFG["acceptance"].items()
                          if k not in ("fp_per_analyst_hour_max", "min_recall_any_class")}}
    g = opv.evaluate_gates(_good(), cfg, SERVING_CFG, _classes(min_recall=0.01))
    assert g["fp_pass"] and g["class_coverage_pass"] and g["all_pass"]


def test_nan_metrics_fail_rather_than_pass_silently():
    g = opv.evaluate_gates(_good(roc_auc=float("nan")), MODEL_CFG, SERVING_CFG, _classes())
    assert not g["roc_pass"] and not g["all_pass"]
