"""Acceptance-gate tests for Alert Triage (#01).

Encodes the plan §8 gates from configs/model_config.yaml `acceptance`: the
discrimination floors, the false-negative ceiling, the precision floor the
threshold is *fitted* to hold, an overfit guard on the train-vs-test ROC gap, the
alert-volume reduction that makes triage worth doing, the analyst-workload ceiling,
and per-attack-class coverage.

These run against the last training run (outputs/metrics.json) and skip when the
model has not been built, so a fresh clone does not fail for work it has not done.
"""
from src.evaluation import operating_point_validator as opv


def test_gates_recorded_by_the_run_all_pass(metrics):
    """The run computes its own verdict — it must be a pass, and it must be complete."""
    g = metrics["acceptance"]
    assert g["all_pass"] is True, f"acceptance failed: {g}"


def test_gates_recompute_from_metrics(metrics, cfg):
    """Re-derive the gates from the recorded metrics: the stored verdict must not
    have been written by a different code path than the validator."""
    g = opv.evaluate_gates(metrics, cfg["model"], cfg["serving"], metrics["per_class"])
    assert g["all_pass"] == metrics["acceptance"]["all_pass"]
    assert g["roc_pass"] and g["pr_pass"] and g["fn_pass"]
    assert g["precision_pass"] and g["not_overfit_pass"] and g["reduction_pass"]
    assert g["fp_pass"] and g["class_coverage_pass"]


def test_roc_auc_meets_floor(metrics, cfg):
    gate = cfg["model"]["acceptance"]["roc_auc_min"]
    assert metrics["roc_auc"] >= gate


def test_pr_auc_meets_floor(metrics, cfg):
    gate = cfg["model"]["acceptance"]["pr_auc_min"]
    assert metrics["pr_auc"] >= gate


def test_fn_rate_under_ceiling(metrics, cfg):
    """Missed attacks are the expensive error — cap them at the operating point."""
    gate = cfg["model"]["acceptance"]["fn_rate_max"]
    assert metrics["fn_rate_%"] / 100.0 <= gate


def test_precision_holds_the_fitted_floor(metrics, cfg):
    """The threshold is fitted to hold this floor; if it does not, the fit failed."""
    assert metrics["precision"] >= cfg["model"]["acceptance"]["precision_min"]


def test_threshold_is_fitted_not_the_fallback(metrics, cfg):
    assert metrics["threshold"] != cfg["model"]["operating_point"]["fallback_threshold"]
    assert 0.0 < metrics["threshold"] < 1.0


def test_not_overfit(metrics, cfg):
    assert metrics["roc_gap"] <= cfg["model"]["acceptance"]["roc_gap_max"]


def test_alert_volume_reduction(metrics, cfg):
    """Triage has to shrink the queue, or it buys the analyst nothing."""
    gate = cfg["model"]["acceptance"]["alert_reduction_min"]
    assert metrics["alert_reduction_%"] / 100.0 >= gate


def test_analyst_workload_ceiling(metrics, cfg):
    fp_hr = opv.fp_per_analyst_hour(metrics, cfg["serving"])
    assert fp_hr <= cfg["model"]["acceptance"]["fp_per_analyst_hour_max"]


def test_every_attack_class_clears_its_floor(metrics, cfg):
    """A healthy aggregate can hide a class the model never catches."""
    floor = cfg["model"]["acceptance"]["min_recall_any_class"]
    weak = [c for c in metrics["per_class"] if c["recall"] < floor]
    assert not weak, f"attack classes below the {floor} recall floor: {weak}"


def test_confusion_matrix_totals_match_test_rows(metrics):
    cells = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    assert cells == metrics["rows_test"]


def test_ranking_quality_at_the_top_of_the_queue(metrics):
    assert metrics["ndcg_at_100"] >= 0.90
    assert metrics["lift_at_1000"] > 1.0, "top-1000 no better than random"
