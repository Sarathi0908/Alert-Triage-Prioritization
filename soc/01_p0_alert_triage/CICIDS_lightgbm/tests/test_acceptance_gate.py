"""Acceptance-gate tests for Alert Triage (#01, CSE-CIC-IDS2018).

Encodes the plan §8 gates from configs/model_config.yaml `acceptance` as tests:
discrimination floors, the false-negative ceiling, the precision floor the
operating point is fitted to, an overfit guard on the train-vs-test ROC gap, and
the alert-volume reduction that makes triage worth doing at all.

These run against the LAST TRAINING RUN (outputs/metrics.json) and skip when the
model has not been built yet, so a fresh clone does not report failures for work
it has not done.
"""


def test_roc_auc_meets_floor(metrics, cfg):
    gate = cfg["model"]["acceptance"]["roc_auc_min"]
    assert metrics["roc_auc"] >= gate, (
        f"ROC-AUC {metrics['roc_auc']} below the {gate} gate")


def test_pr_auc_meets_floor(metrics, cfg):
    gate = cfg["model"]["acceptance"]["pr_auc_min"]
    assert metrics["pr_auc"] >= gate, (
        f"PR-AUC {metrics['pr_auc']} below the {gate} gate")


def test_fn_rate_under_ceiling(metrics, cfg):
    """Missed attacks are the expensive error — cap them at the operating point."""
    gate = cfg["model"]["acceptance"]["fn_rate_max"]
    fn_rate = metrics["fn_rate_%"] / 100.0
    assert fn_rate <= gate, f"FN rate {fn_rate:.4f} above the {gate} ceiling"


def test_precision_holds_operating_point(metrics, cfg):
    """The threshold is *fitted* to hold this floor; if it does not, the fit failed."""
    floor = cfg["model"]["acceptance"]["precision_min"]
    assert metrics["precision"] >= floor, (
        f"precision {metrics['precision']} below the {floor} floor the "
        f"threshold was fitted to hold")


def test_threshold_is_fitted_not_default(metrics, cfg):
    """A threshold equal to the fallback means no point reached the precision floor."""
    assert metrics["threshold"] != cfg["model"]["operating_point"]["fallback_threshold"], (
        "threshold fell back to the default — the precision floor was never met")
    assert 0.0 < metrics["threshold"] < 1.0


def test_not_overfit(metrics, cfg):
    """Train ROC minus test ROC — a widening gap is memorisation."""
    gate = cfg["model"]["acceptance"]["roc_gap_max"]
    assert metrics["roc_gap"] <= gate, (
        f"train-vs-test ROC gap {metrics['roc_gap']} above the {gate} ceiling")


def test_alert_volume_reduction(metrics, cfg):
    """Triage has to actually shrink the queue, or it buys the analyst nothing."""
    gate = cfg["model"]["acceptance"]["alert_reduction_min"]
    reduction = metrics["alert_reduction_%"] / 100.0
    assert reduction >= gate, (
        f"alert reduction {reduction:.4f} below the {gate} gate")


def test_confusion_matrix_totals_match_test_rows(metrics):
    """Internal consistency: the four cells must account for every test row."""
    cells = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    assert cells == metrics["rows_test"]


def test_every_attack_class_is_evaluated(metrics):
    """Per-class recall is the guard against a healthy aggregate hiding a blind spot."""
    per_class = metrics.get("per_class", [])
    assert per_class, "no per-class recall recorded"
    assert all(c["n"] > 0 for c in per_class)
    assert all(0.0 <= c["recall"] <= 1.0 for c in per_class)


def test_ranking_quality_at_the_top_of_the_queue(metrics):
    """Analysts work the top of the queue — it must be dense with real attacks."""
    assert metrics["ndcg_at_100"] >= 0.90
    assert metrics["lift_at_1000"] > 1.0, "top-1000 no better than random"
