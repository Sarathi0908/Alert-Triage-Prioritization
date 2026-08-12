"""
eda_report.py  —  EDA for the merged NetFlow table, with a graph PER ATTACK SCENARIO.

Explores ALL native columns; the model itself uses only the leak/noise-screened
subset (data/native_selected.json), so this is where you see what was left out
and why that was safe.

Writes to outputs/eda/:
  alert_classes.png         count of each attack class (each scenario as a bar)
  per_class_signature.png   heatmap: each class's fingerprint over the model features
  correlation.png           correlation of the model features
  single_feature_auc.png    leak screen (each feature's ROC-AUC alone)
  distributions.png         attack rate by destination port + class share
  column_profile.csv, all_columns_profile.csv, eda_summary.json

Run:  py src/evaluation/eda_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import EDA_DIR, load_config  # noqa: E402
from src import data_source as ds  # noqa: E402
from src.evaluation import plots as pl  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def profile(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Counts, not percentages: nulls, distinct values, duplicated values, range."""
    out = []
    for c in cols:
        s = frame[c]
        v = s.value_counts(dropna=True)
        num = pd.api.types.is_numeric_dtype(s)
        out.append({"column": c, "null_count": int(s.isna().sum()),
                    "unique_count": int(s.nunique()),
                    "duplicate_count": int((v > 1).sum()),
                    "max_repeat": int(v.iloc[0]) if len(v) else 0,
                    "min": round(float(s.min()), 3) if num else "",
                    "max": round(float(s.max()), 3) if num else "",
                    "mean": round(float(s.mean()), 3) if num else ""})
    return pd.DataFrame(out)


def main() -> None:
    mcfg, fcfg = load_config("model"), load_config("feature")
    ecfg = fcfg["eda"]
    seed, target = mcfg["seed"], mcfg["target"]
    cls_col = mcfg["attack_class_column"]
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = ds.load_manifest(mcfg)
    native = list(manifest["native_features"])
    selected = ds.load_selected(mcfg)["features"]

    print("loading merged parquet ...", flush=True)
    df = ds.load_frame(mcfg, columns=native)
    n_atk = int(df[target].sum())
    print(f"rows={len(df):,} attack={n_atk:,} ({df[target].mean() * 100:.3f}%)")

    # ---- 1) each attack scenario as a bar ---------------------------------------
    vc = df[df[cls_col].str.upper() != "BENIGN"][cls_col].value_counts()
    fig, ax = pl.plt.subplots(figsize=(9, 6))
    ax.barh(vc.index[::-1], vc.values[::-1], color=pl.RED, height=.72)
    for i, v in enumerate(vc.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=pl.INK2)
    ax.set_xscale("log")
    pl.style(ax, f"Each attack scenario in {manifest.get('dataset', 'the dataset')} "
                 f"(count, log scale)", "flows")
    pl.save(fig, EDA_DIR / "alert_classes.png")

    # ---- 2) per-class SIGNATURE heatmap -----------------------------------------
    clip = ecfg["zscore_clip"]
    classes = ["BENIGN"] + list(vc.index)
    mu = df.groupby(cls_col)[selected].mean()
    mu = mu.reindex([c for c in classes if c in mu.index])
    z = ((mu - df[selected].mean()) / (df[selected].std() + 1e-9)).clip(-clip, clip)
    fig, ax = pl.plt.subplots(figsize=(max(10, 0.5 * len(selected)), 0.5 * len(z) + 2))
    im = ax.imshow(z.values, cmap=pl.DIV_CMAP, vmin=-clip, vmax=clip, aspect="auto")
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, fontsize=8)
    fig.colorbar(im, shrink=.7, label="std devs from overall mean")
    ax.grid(False)
    pl.style(ax, "Attack-scenario fingerprints — mean of each model feature per class (z-scored)")
    pl.save(fig, EDA_DIR / "per_class_signature.png")

    # ---- 3) correlation of the model features -----------------------------------
    smp = df[selected + [target]].sample(min(ecfg["corr_sample_rows"], len(df)), random_state=seed)
    corr = smp.corr()
    fig, ax = pl.plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap=pl.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=6.5)
    fig.colorbar(im, shrink=.8)
    ax.grid(False)
    pl.style(ax, "Feature correlation — the model features")
    pl.save(fig, EDA_DIR / "correlation.png")
    off = corr.drop(columns=target).drop(index=target).abs().values
    mean_corr = float(off[np.triu_indices_from(off, 1)].mean())

    # ---- 4) single-feature leak screen ------------------------------------------
    ys = smp[target].values
    sep = pd.Series({c: round(max((a := roc_auc_score(ys, smp[c].values)), 1 - a), 4)
                     for c in selected}).sort_values()
    leak_floor = fcfg["screen"]["leak"]["single_feature_auc_min"]
    fig, ax = pl.plt.subplots(figsize=(8.6, 7))
    ax.barh(sep.index, sep.values, height=.7,
            color=[pl.RED if v > 0.9 else pl.BLUE for v in sep.values])
    ax.axvline(0.5, color=pl.MUTED, ls="--", lw=1)
    ax.set_xlim(0.45, 1.0)
    for i, v in enumerate(sep.values):
        ax.text(v + .003, i, f"{v:.2f}", va="center", fontsize=7, color=pl.INK2)
    verdict = "none is a leak" if sep.max() < leak_floor else "LEAK PRESENT"
    pl.style(ax, f"Single-feature separation (leak screen) — {verdict}",
             "ROC-AUC of the feature ALONE")
    pl.save(fig, EDA_DIR / "single_feature_auc.png")

    # ---- 5) distributions: attack rate by top destination ports ------------------
    dp = ecfg["port_column"]
    fig, axes = pl.plt.subplots(1, 2, figsize=(15, 4.6))
    if dp in df.columns:
        top = df[dp].value_counts().head(ecfg["top_ports"]).index
        r = (df[df[dp].isin(top)].groupby(dp)
             .agg(n=(target, "size"), atk=(target, "mean")).reset_index())
        axes[0].bar(r[dp].astype(str), r.atk * 100, color=pl.AQUA, width=.7)
        axes[0].tick_params(axis="x", rotation=60, labelsize=7)
        pl.style(axes[0], "Attack rate by top destination port", "port", "% attack")
    else:
        axes[0].set_axis_off()
        print(f"   note: {dp} not in the table — port panel skipped")
    axes[1].bar(range(len(vc)), vc.values, color=pl.RED, width=.7)
    axes[1].set_xticks(range(len(vc)))
    axes[1].set_xticklabels(vc.index, rotation=70, fontsize=6.5)
    axes[1].set_yscale("log")
    pl.style(axes[1], "Attack-class counts (log)", None, "flows")
    pl.save(fig, EDA_DIR / "distributions.png")

    # ---- profiles ---------------------------------------------------------------
    prof = profile(df, selected + [target])
    prof.to_csv(EDA_DIR / "column_profile.csv", index=False)
    allp = profile(df, native)
    allp["in_model"] = allp.column.isin(selected).map({True: "YES", False: ""})
    allp.to_csv(EDA_DIR / "all_columns_profile.csv", index=False)

    with open(EDA_DIR / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"dataset": manifest.get("dataset"), "rows": int(len(df)),
                   "attack_flows": n_atk,
                   "attack_rate_pct": round(float(df[target].mean() * 100), 4),
                   "attack_classes": vc.to_dict(), "model_features": selected,
                   "mean_abs_feature_correlation": round(mean_corr, 4),
                   "single_feature_auc": sep.sort_values(ascending=False).round(4).to_dict(),
                   "column_profile": prof.to_dict("records")}, fh, indent=2, default=str)

    print(f"mean |feature corr| = {mean_corr:.4f}")
    print("wrote alert_classes.png, per_class_signature.png, correlation.png, "
          "single_feature_auc.png,")
    print("      distributions.png, column_profile.csv, all_columns_profile.csv, eda_summary.json")


if __name__ == "__main__":
    main()
