"""
plots.py  —  evaluation charts for Alert Triage (#01), driven by the plan's §5/§8
protocol. Every function is best-effort: a plotting failure logs and returns None
rather than breaking the training run.

Produces:
  roc_curve.png            — discrimination on the held-out split, with AUC.
  pr_curve.png             — precision/recall against the base rate (imbalanced).
  confusion_matrix.png     — the four outcomes at the fitted operating point.
  calibration.png          — is the probability trustworthy as a probability?
  decile_priority.png      — mean priority per decile over PREDICTED ALERTS ONLY.
  feature_importance.png   — split-gain share per native feature.
  shap_global.png          — mean |SHAP| across alerts (why the model fires at all).
  shap_local.png           — one alert explained (why THIS flow scored).
  per_class_recall.png     — recall per attack class; the blind-spot check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- palette (one visual system across every PNG in outputs/) ----------------
BLUE = "#2a78d6"      # the model / primary series
RED = "#d03b3b"       # attacks, priority
AQUA = "#1baf7a"      # secondary series
INK = "#0b0b0b"       # titles
INK2 = "#52514e"      # axis labels, annotations
MUTED = "#898781"     # ticks, reference lines
SURF = "#fcfcfb"      # figure/axes background
EDGE = "#c3c2b7"
GRID = "#e1e0d9"

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#2a78d6", "#f0efec", "#e34948"]

_RC = {"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
       "axes.edgecolor": EDGE, "xtick.color": MUTED, "ytick.color": MUTED,
       "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False}


def _mpl():
    """Agg selected before pyplot, so these run headless (CI, over SSH)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RC)
    return plt


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def seq_cmap():
    return _cmap(SEQ, "triage_seq")


def div_cmap():
    return _cmap(DIV, "triage_div")


def style(ax, t=None, xl=None, yl=None):
    if t:
        ax.set_title(t, fontsize=12, fontweight="bold", color=INK, pad=12)
    if xl:
        ax.set_xlabel(xl, color=INK2)
    if yl:
        ax.set_ylabel(yl, color=INK2)
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def roc_curve_plot(y_true, scores, out, test_frac=0.30, log=print):
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        plt = _mpl()
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        fig, ax = plt.subplots(figsize=(6.4, 5))
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        style(ax, f"ROC Curve — held-out {round(test_frac * 100)}%",
              "FPR = FP/(FP+TN)", "TPR = TP/(TP+FN)")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot roc_curve failed: {exc}")
        return None


def pr_curve_plot(y_true, scores, out, log=print):
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        plt = _mpl()
        prc, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        base = float(np.asarray(y_true).mean())
        fig, ax = plt.subplots(figsize=(6.4, 5))
        ax.plot(rec, prc, color=BLUE, lw=2, label=f"PR-AUC = {ap:.4f}")
        ax.axhline(base, color=MUTED, lw=1, ls="--", label=f"base rate ({base * 100:.1f}%)")
        style(ax, "Precision–Recall Curve", "Recall", "Precision")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot pr_curve failed: {exc}")
        return None


def confusion_matrix_plot(metrics: dict, out, min_precision=0.70, log=print):
    try:
        plt = _mpl()
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.imshow(cm, cmap=seq_cmap())
        lbl = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{lbl[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * .5 else INK)
        ax.set_xticks([0, 1], ["pred NORMAL", "pred ATTACK"])
        ax.set_yticks([0, 1], ["true NORMAL", "true ATTACK"])
        ax.grid(False)
        style(ax, f"Confusion Matrix — @ ≥{min_precision * 100:.0f}% precision")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot confusion_matrix failed: {exc}")
        return None


def calibration_plot(y_true, scores, out, log=print):
    try:
        from sklearn.calibration import calibration_curve
        plt = _mpl()
        pt, pp = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5.6, 5))
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfect")
        ax.plot(pp, pt, "o-", color=BLUE, lw=2, ms=6, label="model")
        style(ax, "Calibration", "predicted prob", "observed freq")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot calibration failed: {exc}")
        return None


def decile_priority_plot(agg: pd.DataFrame, n_alerts: int, out, n_deciles=10, log=print):
    try:
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(agg.decile, agg.mean_priority, color=RED, width=0.68)
        ax.axhline(float(agg.mean_priority.mean()), color=INK2, ls="--", lw=1.4,
                   label=f"avg ({agg.mean_priority.mean():.0f})")
        ax.set_xticks(range(1, n_deciles + 1))
        style(ax, f"Alert Prioritization — {n_alerts:,} ALERTS only",
              f"decile ({n_deciles}=act first)", "mean priority")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot decile_priority failed: {exc}")
        return None


def _hbar_pct(series: pd.Series, title: str, xlabel: str, out, plt):
    fig, ax = plt.subplots(figsize=(8.4, max(4, 0.4 * len(series))))
    ax.barh(series.index, series.values, color=BLUE, height=0.7)
    for i, v in enumerate(series.values):
        ax.text(v + .2, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    style(ax, title, xlabel)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def feature_importance_plot(gain_pct: pd.Series, out, log=print):
    try:
        plt = _mpl()
        s = gain_pct.sort_values()
        return _hbar_pct(s, f"Feature Importance — {len(s)} native features",
                         "% of total gain", out, plt)
    except Exception as exc:
        log(f"plot feature_importance failed: {exc}")
        return None


def shap_global_plot(shap_values: np.ndarray, feature_names: list[str], out, log=print):
    try:
        plt = _mpl()
        gl = pd.Series(np.abs(shap_values).mean(0), index=feature_names)
        gl = (gl / gl.sum() * 100).sort_values()
        _hbar_pct(gl, "SHAP GLOBAL — mean |impact| across alerts",
                  "mean |SHAP| (%)", out, plt)
        return out, gl.sort_values(ascending=False).round(3).to_dict()
    except Exception as exc:
        log(f"plot shap_global failed: {exc}")
        return None, {}


def shap_local_plot(contribution: np.ndarray, feature_names: list[str], prob: float,
                    out, log=print):
    try:
        plt = _mpl()
        con = pd.Series(contribution, index=feature_names).sort_values(key=abs)
        fig, ax = plt.subplots(figsize=(8.4, max(4, 0.4 * len(con))))
        ax.barh(con.index, con.values, height=0.7,
                color=["#e34948" if v > 0 else BLUE for v in con.values])
        ax.axvline(0, color=EDGE, lw=1)
        style(ax, f"SHAP LOCAL — why THIS alert scored {prob:.2f}",
              "<- NORMAL   SHAP   ATTACK ->")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot shap_local failed: {exc}")
        return None


def per_class_recall_plot(per_class: list[dict], out, log=print):
    try:
        plt = _mpl()
        pc = pd.DataFrame(per_class).sort_values("recall")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([s[:34] for s in pc.attack_class.astype(str)], pc.recall,
                color=BLUE, height=0.68)
        for i, (_, r) in enumerate(pc.iterrows()):
            ax.text(r.recall + .01, i, f"{r.recall:.2f} (n={r.n:,})", va="center",
                    fontsize=8, color=INK2)
        ax.set_xlim(0, 1.25)
        style(ax, "Detection recall per attack class", "recall")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot per_class_recall failed: {exc}")
        return None
