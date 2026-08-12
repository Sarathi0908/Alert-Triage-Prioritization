"""
flow_traffic_profiler.py  —  EDA over the merged NetFlow table (plan §3).

Profiles ALL native columns, not just the ones the model keeps, so the screen's
decisions are auditable: you can see what was dropped and confirm it was safe.
Reusable logic lives here; notebooks/ only explores (instruction §3-§4).

Writes to outputs/eda/:
  alert_classes.png         count of each attack class (each campaign as a bar)
  per_class_signature.png   heatmap — each class's fingerprint over the model features
  correlation.png           correlation of the model features
  single_feature_auc.png    leak screen — each feature's ROC-AUC alone
  distributions.png         attack rate by destination port + class share
  column_profile.csv        counts for the model features
  all_columns_profile.csv   counts for every native column, flagged in/out of model
  eda_summary.json          the numbers behind the charts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation import plots as pl


def profile(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Counts, not percentages: nulls, distinct values, duplicated values, range."""
    out = []
    for c in cols:
        s = frame[c]
        v = s.value_counts(dropna=True)
        num = pd.api.types.is_numeric_dtype(s)
        # float32 counters can overflow when summed across ~1M rows; mean is taken
        # in float64 so the reported statistic stays finite.
        s64 = s.astype("float64") if num else s
        out.append({"column": c, "null_count": int(s.isna().sum()),
                    "unique_count": int(s.nunique()),
                    "duplicate_count": int((v > 1).sum()),
                    "max_repeat": int(v.iloc[0]) if len(v) else 0,
                    "min": round(float(s64.min()), 3) if num else "",
                    "max": round(float(s64.max()), 3) if num else "",
                    "mean": round(float(s64.mean()), 3) if num else ""})
    return pd.DataFrame(out)


def attack_class_counts(df: pd.DataFrame, cls_col: str) -> pd.Series:
    return df[df[cls_col].str.upper() != "BENIGN"][cls_col].value_counts()


def class_signature(df: pd.DataFrame, selected: list[str], cls_col: str,
                    vc: pd.Series, clip: float) -> pd.DataFrame:
    """Each class's mean feature vector, z-scored against the overall mean."""
    classes = ["BENIGN"] + list(vc.index)
    mu = df.groupby(cls_col)[selected].mean()
    mu = mu.reindex([c for c in classes if c in mu.index])
    return ((mu - df[selected].mean()) / (df[selected].std() + 1e-9)).clip(-clip, clip)


def single_feature_auc(sample: pd.DataFrame, selected: list[str], target: str) -> pd.Series:
    """Each feature's ROC-AUC on its own — the leak screen, made visible."""
    ys = sample[target].values
    return pd.Series({c: round(max((a := roc_auc_score(ys, sample[c].values)), 1 - a), 4)
                      for c in selected}).sort_values()


def run(df: pd.DataFrame, selected: list[str], native: list[str], manifest: dict,
        model_cfg: dict, feature_cfg: dict, eda_dir, log=print) -> dict:
    """Produce every EDA artefact. Charts are best-effort; the summary is not."""
    from pathlib import Path
    eda_dir = Path(eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)
    ecfg = feature_cfg["eda"]
    seed, target = model_cfg["seed"], model_cfg["target"]
    cls_col = model_cfg["attack_class_column"]
    clip = ecfg["zscore_clip"]

    n_atk = int(df[target].sum())
    log(f"rows={len(df):,} attack={n_atk:,} ({df[target].mean() * 100:.3f}%)")
    vc = attack_class_counts(df, cls_col)
    plt = pl._mpl()

    # ---- 1) each attack campaign as a bar ---------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(vc.index[::-1], vc.values[::-1], color=pl.RED, height=.72)
        for i, v in enumerate(vc.values[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=pl.INK2)
        ax.set_xscale("log")
        pl.style(ax, f"Each attack scenario in {manifest.get('dataset', 'the dataset')} "
                     f"(count, log scale)", "flows")
        fig.tight_layout(); fig.savefig(eda_dir / "alert_classes.png", dpi=150); plt.close(fig)
    except Exception as exc:
        log(f"plot alert_classes failed: {exc}")

    # ---- 2) per-class fingerprint heatmap ---------------------------------------
    try:
        z = class_signature(df, selected, cls_col, vc, clip)
        fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(selected)), 0.5 * len(z) + 2))
        im = ax.imshow(z.values, cmap=pl.div_cmap(), vmin=-clip, vmax=clip, aspect="auto")
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels(selected, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(z)))
        ax.set_yticklabels(z.index, fontsize=8)
        fig.colorbar(im, shrink=.7, label="std devs from overall mean")
        ax.grid(False)
        pl.style(ax, "Attack-scenario fingerprints — mean of each model feature "
                     "per class (z-scored)")
        fig.tight_layout(); fig.savefig(eda_dir / "per_class_signature.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        log(f"plot per_class_signature failed: {exc}")

    # ---- 3) correlation of the model features -----------------------------------
    smp = df[selected + [target]].sample(min(ecfg["corr_sample_rows"], len(df)),
                                         random_state=seed)
    corr = smp.corr()
    off = corr.drop(columns=target).drop(index=target).abs().values
    mean_corr = float(off[np.triu_indices_from(off, 1)].mean())
    try:
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(corr, cmap=pl.div_cmap(), vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns, fontsize=6.5)
        fig.colorbar(im, shrink=.8)
        ax.grid(False)
        pl.style(ax, "Feature correlation — the model features")
        fig.tight_layout(); fig.savefig(eda_dir / "correlation.png", dpi=150); plt.close(fig)
    except Exception as exc:
        log(f"plot correlation failed: {exc}")

    # ---- 4) single-feature leak screen ------------------------------------------
    sep = single_feature_auc(smp, selected, target)
    leak_floor = feature_cfg["screen"]["leak"]["single_feature_auc_min"]
    try:
        fig, ax = plt.subplots(figsize=(8.6, 7))
        ax.barh(sep.index, sep.values, height=.7,
                color=[pl.RED if v > 0.9 else pl.BLUE for v in sep.values])
        ax.axvline(0.5, color=pl.MUTED, ls="--", lw=1)
        ax.set_xlim(0.45, 1.0)
        for i, v in enumerate(sep.values):
            ax.text(v + .003, i, f"{v:.2f}", va="center", fontsize=7, color=pl.INK2)
        verdict = "none is a leak" if sep.max() < leak_floor else "LEAK PRESENT"
        pl.style(ax, f"Single-feature separation (leak screen) — {verdict}",
                 "ROC-AUC of the feature ALONE")
        fig.tight_layout(); fig.savefig(eda_dir / "single_feature_auc.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        log(f"plot single_feature_auc failed: {exc}")

    # ---- 5) attack rate by top destination ports --------------------------------
    dp = ecfg["port_column"]
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
        if dp in df.columns:
            top = df[dp].value_counts().head(ecfg["top_ports"]).index
            r = (df[df[dp].isin(top)].groupby(dp)
                 .agg(n=(target, "size"), atk=(target, "mean")).reset_index())
            axes[0].bar(r[dp].astype(str), r.atk * 100, color=pl.AQUA, width=.7)
            axes[0].tick_params(axis="x", rotation=60, labelsize=7)
            pl.style(axes[0], "Attack rate by top destination port", "port", "% attack")
        else:
            axes[0].set_axis_off()
            log(f"   note: {dp} not in the table — port panel skipped")
        axes[1].bar(range(len(vc)), vc.values, color=pl.RED, width=.7)
        axes[1].set_xticks(range(len(vc)))
        axes[1].set_xticklabels(vc.index, rotation=70, fontsize=6.5)
        axes[1].set_yscale("log")
        pl.style(axes[1], "Attack-class counts (log)", None, "flows")
        fig.tight_layout(); fig.savefig(eda_dir / "distributions.png", dpi=150); plt.close(fig)
    except Exception as exc:
        log(f"plot distributions failed: {exc}")

    # ---- profiles + summary -----------------------------------------------------
    prof = profile(df, selected + [target])
    prof.to_csv(eda_dir / "column_profile.csv", index=False)
    allp = profile(df, native)
    allp["in_model"] = allp.column.isin(selected).map({True: "YES", False: ""})
    allp.to_csv(eda_dir / "all_columns_profile.csv", index=False)

    summary = {"dataset": manifest.get("dataset"), "rows": int(len(df)),
               "attack_flows": n_atk,
               "attack_rate_pct": round(float(df[target].mean() * 100), 4),
               "attack_classes": vc.to_dict(), "model_features": selected,
               "mean_abs_feature_correlation": round(mean_corr, 4),
               "single_feature_auc": sep.sort_values(ascending=False).round(4).to_dict(),
               "column_profile": prof.to_dict("records")}
    with open(eda_dir / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log(f"mean |feature corr| = {mean_corr:.4f}")
    log("eda -> alert_classes, per_class_signature, correlation, single_feature_auc, "
        "distributions + profiles + eda_summary.json")
    return summary
