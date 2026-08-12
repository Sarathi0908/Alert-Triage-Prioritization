"""Smoke tests for src/evaluation/plots.py — charts are best-effort, so we assert
they produce a file (or return None instead of raising) on toy data.
"""
import numpy as np
import pandas as pd

from src.evaluation import plots as pl

_QUIET = lambda *_: None          # noqa: E731 — the log sink these functions take


def _labels(n=200):
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    scores = np.r_[np.linspace(0.01, 0.45, n // 2), np.linspace(0.55, 0.99, n // 2)]
    return y, scores


def test_roc_curve_writes_a_file(tmp_path):
    y, s = _labels()
    out = tmp_path / "roc.png"
    assert pl.roc_curve_plot(y, s, str(out), log=_QUIET)
    assert out.exists()


def test_pr_curve_writes_a_file(tmp_path):
    y, s = _labels()
    out = tmp_path / "pr.png"
    assert pl.pr_curve_plot(y, s, str(out), log=_QUIET)
    assert out.exists()


def test_confusion_matrix_writes_a_file(tmp_path):
    out = tmp_path / "cm.png"
    assert pl.confusion_matrix_plot({"tn": 90, "fp": 10, "fn": 5, "tp": 95},
                                    str(out), log=_QUIET)
    assert out.exists()


def test_calibration_writes_a_file(tmp_path):
    y, s = _labels()
    out = tmp_path / "cal.png"
    assert pl.calibration_plot(y, s, str(out), log=_QUIET)
    assert out.exists()


def test_decile_priority_writes_a_file(tmp_path):
    agg = pd.DataFrame({"decile": range(1, 11),
                        "mean_priority": np.linspace(10, 95, 10)})
    out = tmp_path / "dec.png"
    assert pl.decile_priority_plot(agg, 1234, str(out), log=_QUIET)
    assert out.exists()


def test_feature_importance_writes_a_file(tmp_path):
    out = tmp_path / "imp.png"
    gain = pd.Series({"L4_DST_PORT": 40.6, "MIN_TTL": 4.8})
    assert pl.feature_importance_plot(gain, str(out), log=_QUIET)
    assert out.exists()


def test_shap_global_returns_file_and_shares(tmp_path):
    out = tmp_path / "shap.png"
    sv = np.array([[1.0, -2.0], [0.5, -1.5]])
    png, shares = pl.shap_global_plot(sv, ["a", "b"], str(out), log=_QUIET)
    assert png and out.exists()
    assert set(shares) == {"a", "b"}
    assert abs(sum(shares.values()) - 100.0) < 1e-6


def test_shap_local_writes_a_file(tmp_path):
    out = tmp_path / "local.png"
    assert pl.shap_local_plot(np.array([1.2, -0.4]), ["a", "b"], 0.93, str(out), log=_QUIET)
    assert out.exists()


def test_per_class_recall_writes_a_file(tmp_path):
    out = tmp_path / "pcr.png"
    per_class = [{"attack_class": "DoS", "n": 100, "recall": 0.99},
                 {"attack_class": "Bot", "n": 10, "recall": 0.4}]
    assert pl.per_class_recall_plot(per_class, str(out), log=_QUIET)
    assert out.exists()


def test_a_plot_failure_returns_none_instead_of_raising(tmp_path):
    """Charts must never take down a training run — bad input logs and returns None."""
    assert pl.per_class_recall_plot([], str(tmp_path / "x.png"), log=_QUIET) is None
    assert pl.confusion_matrix_plot({}, str(tmp_path / "y.png"), log=_QUIET) is None


def test_palette_and_cmaps_are_available():
    assert pl.seq_cmap() is not None and pl.div_cmap() is not None
    assert pl.BLUE.startswith("#") and pl.RED.startswith("#")
