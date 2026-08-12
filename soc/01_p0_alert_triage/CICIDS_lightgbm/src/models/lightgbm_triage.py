"""
lightgbm_triage.py  —  THE Alert Triage model. LightGBM, NATIVE-only features.

X = the leak/noise-screened native NetFlow features (data/native_selected.json).
Y = y_is_attack (BENIGN=0, any attack=1).
Split = random stratified hold-out (see configs/model_config.yaml `split`).

Removes the previous run's outputs (EDA charts are preserved), then regenerates
the model bundle + metrics + ROC/PR/confusion/calibration/decile/feature-importance/
SHAP charts + per-attack-class recall.

Every threshold and hyperparameter comes from configs/model_config.yaml.
Run:  py src/models/lightgbm_triage.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import EDA_DIR, OUTPUT_DIR, load_config  # noqa: E402
from src import data_source as ds  # noqa: E402
from src.evaluation import plots as pl  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def clean_outputs() -> None:
    """Drop the last run's artefacts but keep outputs/eda/ (owned by the EDA step)."""
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.rglob("*"):
            if p.is_file() and p.parent.name != "eda":
                try:
                    p.unlink()
                except OSError:
                    pass
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k(y_true, scores, k: int) -> float:
    return float(np.asarray(y_true)[np.argsort(-scores)[:k]].mean())


def ndcg_at_k(y_true, scores, k: int) -> float:
    gains = np.asarray(y_true)[np.argsort(-scores)[:k]]
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(np.asarray(y_true))[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def pick_threshold(y_true, scores, min_precision: float, fallback: float) -> float:
    """Highest-recall threshold that still holds precision >= the floor."""
    prc, rec, thr = precision_recall_curve(y_true, scores)
    ok = np.where(prc[:-1] >= min_precision)[0]
    if not len(ok):
        return float(fallback)
    return float(thr[ok[np.argmax(rec[:-1][ok])]])


def main() -> None:
    mcfg = load_config("model")
    seed = mcfg["seed"]
    op = mcfg["operating_point"]
    pri_cfg = mcfg["priority"]

    print("0) cleaning outputs")
    clean_outputs()

    # ---- 1) features ------------------------------------------------------------
    sel = ds.load_selected(mcfg)
    features, target = sel["features"], mcfg["target"]
    cls_col = mcfg["attack_class_column"]
    vol_feature = ds.pick_volume_feature(features, pri_cfg["volume_feature_preference"])
    print(f"1) {len(features)} native features · volume term = {vol_feature}")

    df = ds.load_frame(mcfg, columns=features)
    y = df[target].astype(int).values
    print(f"   rows={len(df):,}  attack={y.mean() * 100:.3f}%")

    # ---- 2) split + fit ---------------------------------------------------------
    tr, te = train_test_split(np.arange(len(df)), test_size=mcfg["split"]["test_frac"],
                              random_state=seed,
                              stratify=y if mcfg["split"]["stratify"] else None)
    train, test = df.iloc[tr], df.iloc[te]
    Xtr, ytr = train[features], y[tr]
    Xte, yte = test[features], y[te]
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    clf = lgb.LGBMClassifier(scale_pos_weight=spw, random_state=seed, **mcfg["lgbm_classifier"])
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    ptr = clf.predict_proba(Xtr)[:, 1]
    roc, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
    tr_roc = roc_auc_score(ytr, ptr)
    prc, rec, _ = precision_recall_curve(yte, p)
    thr_value = pick_threshold(yte, p, op["min_precision"], op["fallback_threshold"])
    pred = (p >= thr_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    base = float(yte.mean())
    print(f"2) train ROC={tr_roc:.4f} | TEST ROC={roc:.4f} PR={pr:.4f} gap={tr_roc - roc:.4f}")
    print(f"   precision={precision_score(yte, pred):.3f} recall={recall_score(yte, pred):.3f} "
          f"FN%={fn / max(fn + tp, 1) * 100:.3f}")

    M = {
        "model": "CSE-CIC-IDS2018 Alert-Triage (native-only, LightGBM)",
        "dataset": ds.load_manifest(mcfg).get("dataset", "CSE-CIC-IDS2018 (NF-v2, NetFlow)"),
        "rows_total": int(len(df)), "rows_train": int(len(tr)), "rows_test": int(len(te)),
        "attack_rate_%": round(float(y.mean() * 100), 4), "base_rate_%": round(base * 100, 4),
        "n_features": len(features), "features": features,
        "protocol": (f"random {round((1 - mcfg['split']['test_frac']) * 100)}/"
                     f"{round(mcfg['split']['test_frac'] * 100)} split "
                     f"({'stratified, ' if mcfg['split']['stratify'] else ''}seed={seed})"),
        "train_roc": round(float(tr_roc), 4), "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4), "roc_gap": round(float(tr_roc - roc), 4),
        "threshold": round(thr_value, 6), "min_precision_floor": op["min_precision"],
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(yte, pred)), 4),
        "brier": round(float(brier_score_loss(yte, p)), 6),
        "log_loss": round(float(log_loss(yte, p, labels=[0, 1])), 6),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fn_rate_%": round(float(fn / max(fn + tp, 1) * 100), 3),
        "ndcg_at_100": round(ndcg_at_k(yte, p, 100), 4),
        "precision_at_1000": round(precision_at_k(yte, p, 1000), 4),
        "lift_at_1000": round(precision_at_k(yte, p, 1000) / max(base, 1e-9), 2),
        "alert_reduction_%": round(float((1 - pred.mean()) * 100), 2),
        "hyperparameters": {k: mcfg["lgbm_classifier"][k] for k in
                            ("n_estimators", "num_leaves", "min_child_samples", "learning_rate")
                            if k in mcfg["lgbm_classifier"]} | {"scale_pos_weight": round(spw, 1)},
    }

    # ---- per-attack-class recall ------------------------------------------------
    tt = test.assign(_pred=pred)
    attacks = tt[tt[cls_col].str.upper() != "BENIGN"]
    M["per_class"] = [{"attack_class": a, "n": int(len(d)),
                       "recall": round(float((d._pred == 1).mean()), 4)}
                      for a, d in attacks.groupby(cls_col)]
    per_class = pd.DataFrame(M["per_class"]).sort_values("n", ascending=False)
    print("\n   PER-CLASS RECALL:")
    print(per_class.to_string(index=False))
    per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    # ---- charts -----------------------------------------------------------------
    fpr, tpr, _ = roc_curve(yte, p)
    fig, ax = pl.plt.subplots(figsize=(6.4, 5))
    ax.plot(fpr, tpr, color=pl.BLUE, lw=2, label=f"AUC = {roc:.4f}")
    ax.plot([0, 1], [0, 1], color=pl.MUTED, lw=1, ls="--")
    pl.style(ax, f"ROC Curve — held-out {round(mcfg['split']['test_frac'] * 100)}%",
             "FPR = FP/(FP+TN)", "TPR = TP/(TP+FN)")
    ax.legend(frameon=False, loc="lower right")
    pl.save(fig, OUTPUT_DIR / "roc_curve.png")

    fig, ax = pl.plt.subplots(figsize=(6.4, 5))
    ax.plot(rec, prc, color=pl.BLUE, lw=2, label=f"PR-AUC = {pr:.4f}")
    ax.axhline(base, color=pl.MUTED, lw=1, ls="--", label=f"base rate ({base * 100:.1f}%)")
    pl.style(ax, "Precision–Recall Curve", "Recall", "Precision")
    ax.legend(frameon=False)
    pl.save(fig, OUTPUT_DIR / "pr_curve.png")

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = pl.plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap=pl.SEQ_CMAP)
    lbl = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{lbl[i][j]}\n{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() * .5 else pl.INK)
    ax.set_xticks([0, 1], ["pred NORMAL", "pred ATTACK"])
    ax.set_yticks([0, 1], ["true NORMAL", "true ATTACK"])
    ax.grid(False)
    pl.style(ax, f"Confusion Matrix — @ ≥{op['min_precision'] * 100:.0f}% precision")
    pl.save(fig, OUTPUT_DIR / "confusion_matrix.png")

    pt_, pp_ = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    fig, ax = pl.plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=pl.MUTED, ls="--", lw=1, label="perfect")
    ax.plot(pp_, pt_, "o-", color=pl.BLUE, lw=2, ms=6, label="model")
    pl.style(ax, "Calibration", "predicted prob", "observed freq")
    ax.legend(frameon=False)
    pl.save(fig, OUTPUT_DIR / "calibration.png")

    # ---- decile priority (predicted ALERTS only) --------------------------------
    n_dec = pri_cfg["n_deciles"]
    w = pri_cfg["weights"]
    alerts = tt[tt._pred == 1].copy()
    raw_vol = pd.to_numeric(alerts[vol_feature], errors="coerce").fillna(0).values.astype(float)
    vol = np.log1p(raw_vol) if pri_cfg["log1p_volume"] else raw_vol
    pa = p[pred == 1]
    pri = 100.0 * (w["probability"] * pd.Series(pa).rank(pct=True).values
                   + w["volume"] * pd.Series(vol).rank(pct=True).values)
    alerts["priority"] = pri
    alerts["decile"] = pd.qcut(alerts.priority.rank(method="first"), n_dec, labels=False) + 1
    agg = alerts.groupby("decile").agg(mean_priority=("priority", "mean"), n=("priority", "size"),
                                       hits=(target, "sum")).reset_index()
    agg["hit_%"] = (agg.hits / agg.n * 100).round(2)
    fig, ax = pl.plt.subplots(figsize=(9, 5))
    ax.bar(agg.decile, agg.mean_priority, color=pl.RED, width=0.68)
    ax.axhline(float(agg.mean_priority.mean()), color=pl.INK2, ls="--", lw=1.4,
               label=f"avg ({agg.mean_priority.mean():.0f})")
    ax.set_xticks(range(1, n_dec + 1))
    pl.style(ax, f"Alert Prioritization — {len(alerts):,} ALERTS only",
             f"decile ({n_dec}=act first)", "mean priority")
    ax.legend(frameon=False)
    pl.save(fig, OUTPUT_DIR / "decile_priority.png")
    M["decile"] = agg.to_dict("records")
    M["decile_basis"] = {
        "n_alerts": int(len(alerts)),
        "formula": (f"priority = 100*({w['probability']}*pct(prob) + "
                    f"{w['volume']}*pct(log1p({vol_feature})))"),
        "note": "predicted ALERTS only — normal traffic not ranked"}

    # ---- feature importance -----------------------------------------------------
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=features)
    imp = (imp / imp.sum() * 100).sort_values()
    M["gain_pct"] = imp.sort_values(ascending=False).round(3).to_dict()
    fig, ax = pl.plt.subplots(figsize=(8.4, max(4, 0.4 * len(imp))))
    ax.barh(imp.index, imp.values, color=pl.BLUE, height=0.7)
    for i, v in enumerate(imp.values):
        ax.text(v + .2, i, f"{v:.1f}%", va="center", fontsize=8, color=pl.INK2)
    pl.style(ax, f"Feature Importance — {len(features)} native features", "% of total gain")
    pl.save(fig, OUTPUT_DIR / "feature_importance.png")

    # ---- SHAP (LightGBM native pred_contrib — exact tree SHAP, no extra dep) -----
    xcfg = mcfg["explainability"]
    try:
        Xs = Xte.sample(min(xcfg["shap_sample_rows"], len(Xte)), random_state=seed)
        sv = clf.booster_.predict(Xs, pred_contrib=True)[:, :-1]
        gl = pd.Series(np.abs(sv).mean(0), index=features)
        gl = (gl / gl.sum() * 100).sort_values()
        fig, ax = pl.plt.subplots(figsize=(8.4, max(4, 0.4 * len(gl))))
        ax.barh(gl.index, gl.values, color=pl.BLUE, height=0.7)
        for i, v in enumerate(gl.values):
            ax.text(v + .2, i, f"{v:.1f}%", va="center", fontsize=8, color=pl.INK2)
        pl.style(ax, "SHAP GLOBAL — mean |impact| across alerts", "mean |SHAP| (%)")
        pl.save(fig, OUTPUT_DIR / "shap_global.png")

        ps = clf.predict_proba(Xs)[:, 1]
        i = int(np.argmax(ps))
        con = pd.Series(sv[i], index=features).sort_values(key=abs)
        fig, ax = pl.plt.subplots(figsize=(8.4, max(4, 0.4 * len(con))))
        ax.barh(con.index, con.values, color=pl.signed_colors(con.values), height=0.7)
        ax.axvline(0, color=pl.EDGE, lw=1)
        pl.style(ax, f"SHAP LOCAL — why THIS alert scored {ps[i]:.2f}",
                 "<- NORMAL   SHAP   ATTACK ->")
        pl.save(fig, OUTPUT_DIR / "shap_local.png")
        M["shap_pct"] = gl.sort_values(ascending=False).round(3).to_dict()
    except Exception as e:                                    # noqa: BLE001
        print("SHAP skipped:", str(e)[:80])

    pc = per_class.sort_values("recall")
    fig, ax = pl.plt.subplots(figsize=(9, 5))
    ax.barh([s[:34] for s in pc.attack_class.astype(str)], pc.recall, color=pl.BLUE, height=0.68)
    for i, (_, r) in enumerate(pc.iterrows()):
        ax.text(r.recall + .01, i, f"{r.recall:.2f} (n={r.n:,})", va="center", fontsize=8,
                color=pl.INK2)
    ax.set_xlim(0, 1.25)
    pl.style(ax, "Detection recall per attack class", "recall")
    pl.save(fig, OUTPUT_DIR / "per_class_recall.png")

    # ---- bundle + metrics -------------------------------------------------------
    # The threshold and the percentile grids travel WITH the model so the served
    # operating point can never drift from the evaluated one.
    q101 = np.linspace(0, 1, 101)
    joblib.dump({"model": clf, "features": features, "native": features,
                 "cats": [], "cat_levels": {},
                 "threshold": thr_value,
                 "decile_edges": ([float(x) for x in np.quantile(pri, np.linspace(0, 1, n_dec + 1))]
                                  if len(pri) else None),
                 "vol_feature": vol_feature,
                 "p_grid": [float(x) for x in np.quantile(pa, q101)],
                 "vol_grid": [float(x) for x in np.quantile(vol, q101)],
                 "rank_q": [float(x) for x in q101],
                 "priority_weights": dict(w)},
                OUTPUT_DIR / "lgbm_model.pkl")
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(M, fh, indent=2, default=float)
    print("\nSAVED -> outputs/lgbm_model.pkl + metrics.json + charts + per_class_metrics.csv")


if __name__ == "__main__":
    main()
