"""
operating_point_validator.py  —  turn detection metrics into the plan's §8
acceptance gates, expressed in analyst-workload terms.

The operating point is a fitted threshold: the highest-recall point that still
holds the configured precision floor. From the resulting confusion matrix we
derive false-positives-per-analyst-hour (are analysts overwhelmed?) and check the
plan gates: discrimination floors, the false-negative ceiling, the precision floor
the threshold was fitted to hold, an overfit guard on the train-vs-test ROC gap,
and the alert-volume reduction that makes triage worth doing at all.
"""
from __future__ import annotations

import math


def fp_per_analyst_hour(metrics: dict, serving_cfg: dict) -> float:
    """False alerts per hour of analyst review time at the flagged operating point.

    FP alerts = flows queued that were not attacks; each costs
    `analyst_minutes_per_alert` to triage.
    """
    fp = metrics.get("fp", 0)
    flagged = fp + metrics.get("tp", 0)
    minutes = serving_cfg["scoring"].get("analyst_minutes_per_alert", 5)
    analyst_hours = flagged * minutes / 60.0
    return float(fp / analyst_hours) if analyst_hours > 0 else float("nan")


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def evaluate_gates(metrics: dict, model_cfg: dict, serving_cfg: dict,
                   per_class: list[dict] | None = None) -> dict:
    """Apply the §8 acceptance gates to one training run's metrics."""
    acc = model_cfg["acceptance"]
    fitted_floor = model_cfg["operating_point"]["min_precision"]
    fp_hr = fp_per_analyst_hour(metrics, serving_cfg)

    roc = metrics.get("roc_auc", float("nan"))
    pr = metrics.get("pr_auc", float("nan"))
    prec = metrics.get("precision", float("nan"))
    fn_rate = metrics.get("fn_rate_%", float("nan")) / 100.0
    gap = metrics.get("roc_gap", float("nan"))
    reduction = metrics.get("alert_reduction_%", float("nan")) / 100.0

    weakest = (min(per_class, key=lambda r: r["recall"]) if per_class else None)
    min_class_recall = weakest["recall"] if weakest else float("nan")
    class_floor = acc.get("min_recall_any_class")

    gates = {
        "operating_point": f"threshold fitted at >={fitted_floor:.0%} precision",
        "roc_auc": roc, "roc_auc_min": acc["roc_auc_min"],
        "roc_pass": bool(_ok(roc) and roc >= acc["roc_auc_min"]),
        "pr_auc": pr, "pr_auc_min": acc["pr_auc_min"],
        "pr_pass": bool(_ok(pr) and pr >= acc["pr_auc_min"]),
        "fn_rate": fn_rate, "fn_rate_max": acc["fn_rate_max"],
        "fn_pass": bool(_ok(fn_rate) and fn_rate <= acc["fn_rate_max"]),
        "precision": prec, "precision_min": acc["precision_min"],
        "precision_pass": bool(_ok(prec) and prec >= acc["precision_min"]),
        "roc_gap": gap, "roc_gap_max": acc["roc_gap_max"],
        "not_overfit_pass": bool(_ok(gap) and gap <= acc["roc_gap_max"]),
        "alert_reduction": reduction, "alert_reduction_min": acc["alert_reduction_min"],
        "reduction_pass": bool(_ok(reduction) and reduction >= acc["alert_reduction_min"]),
        "fp_per_analyst_hour": fp_hr,
        "fp_per_analyst_hour_max": acc.get("fp_per_analyst_hour_max"),
        "weakest_attack_class": (weakest or {}).get("attack_class"),
        "min_recall_any_class": min_class_recall,
        "min_recall_any_class_floor": class_floor,
    }
    gates["fp_pass"] = bool(
        acc.get("fp_per_analyst_hour_max") is None
        or (_ok(fp_hr) and fp_hr <= acc["fp_per_analyst_hour_max"]))
    gates["class_coverage_pass"] = bool(
        class_floor is None
        or (_ok(min_class_recall) and min_class_recall >= class_floor))

    gates["all_pass"] = bool(gates["roc_pass"] and gates["pr_pass"] and gates["fn_pass"]
                             and gates["precision_pass"] and gates["not_overfit_pass"]
                             and gates["reduction_pass"] and gates["fp_pass"]
                             and gates["class_coverage_pass"])
    return gates
