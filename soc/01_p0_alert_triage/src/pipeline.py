"""
pipeline.py  —  end-to-end Alert Triage pipeline (#01).

run_training():
  ingest merged NetFlow -> leak/noise screen over the native columns -> stratified
  hold-out split -> LightGBM triage classifier -> threshold fitted to the precision
  floor -> priority + decile over predicted ALERTS ONLY -> flow and per-attack-class
  evaluation -> §8 acceptance gates -> persist bundle + metrics + charts + SIEM risk
  index.

All settings come from configs/*.yaml; no magic numbers here.
Run:  py src/pipeline.py            (add --skip-screen to reuse data/native_selected.json)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_SLOT_ROOT = Path(__file__).resolve().parents[1]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import load_all_configs, resolve_path                    # noqa: E402
from src import data_source as ds                                        # noqa: E402
from src.features import native_feature_screener as fs                   # noqa: E402
from src.features.priority_score_builder import PriorityScoreBuilder     # noqa: E402
from src.models.lightgbm_triage import LightGBMTriage                    # noqa: E402
from src.evaluation import attack_class_evaluator as ace                 # noqa: E402
from src.evaluation import operating_point_validator as opv              # noqa: E402
from src.evaluation import flow_traffic_profiler as ftp                  # noqa: E402
from src.evaluation import plots as pl                                   # noqa: E402
from src.serving import siem_platform_connector as sc                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
_t0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


def _clean_outputs(out_dir: Path) -> None:
    """Drop the last run's artefacts but keep outputs/eda/ until the EDA step
    rewrites it, so a failed run does not leave the slot with no EDA at all."""
    if out_dir.exists():
        for p in out_dir.rglob("*"):
            if p.is_file() and p.parent.name != "eda":
                try:
                    p.unlink()
                except OSError:
                    pass
    (out_dir / "eda").mkdir(parents=True, exist_ok=True)


def run_screen(mcfg: dict, fcfg: dict) -> list[str]:
    """Leak/noise/dominance screen over the native columns -> native_selected.json."""
    seed = mcfg["seed"]
    exclude = set(fcfg.get("exclude") or [])
    native = [c for c in ds.native_features(mcfg) if c not in exclude]
    frame = ds.load_frame(mcfg, columns=native, with_class=False)
    sample = ds.subsample(frame, fcfg["screen"]["sample_rows"], seed)
    y = sample[mcfg["target"]].astype(int).values
    _log(f"screening {len(native)} native features on {len(sample):,} rows "
         f"(attack {y.mean() * 100:.2f}%)")
    selected, audit, payload = fs.screen(sample, y, native, fcfg, seed=seed, log=_log)
    audit.to_csv(ds.data_dir(mcfg) / mcfg["data"]["screen_audit"], index=False)
    ds.save_selected(mcfg, payload)
    fs.log_summary(payload, fcfg, log=_log)
    return selected


def run_training(out_dir: str | None = None, skip_screen: bool = False) -> dict:
    cfg = load_all_configs()
    mcfg, fcfg, scfg = cfg["model"], cfg["feature"], cfg["serving"]
    seed = mcfg["seed"]
    op, pri_cfg = mcfg["operating_point"], mcfg["priority"]
    target, cls_col = mcfg["target"], mcfg["attack_class_column"]
    out_dir = Path(out_dir or resolve_path("outputs"))
    os.makedirs(out_dir, exist_ok=True)
    _clean_outputs(out_dir)

    # ----- screen -----------------------------------------------------------
    if skip_screen:
        features = ds.load_selected(mcfg)["features"]
        _log(f"reusing screened feature set ({len(features)} features)")
    else:
        features = run_screen(mcfg, fcfg)

    # ----- ingest -----------------------------------------------------------
    data = ds.load_dataset(mcfg, columns=features, log=_log)
    df = data.frame
    y = df[target].astype(int).values
    vol_feature = ds.pick_volume_feature(features, pri_cfg["volume_feature_preference"])
    _log(f"{len(features)} native features · volume term = {vol_feature}")

    # ----- split ------------------------------------------------------------
    tr, te = train_test_split(np.arange(len(df)), test_size=mcfg["split"]["test_frac"],
                              random_state=seed,
                              stratify=y if mcfg["split"]["stratify"] else None)
    train, test = df.iloc[tr], df.iloc[te]
    ytr, yte = y[tr], y[te]
    _log(f"split=random stratified {round((1 - mcfg['split']['test_frac']) * 100)}/"
         f"{round(mcfg['split']['test_frac'] * 100)} train={len(tr):,} test={len(te):,}")

    # ----- fit + fitted operating point ------------------------------------
    model = LightGBMTriage(mcfg["lgbm_classifier"], seed).fit(train[features], features, ytr)
    p = model.attack_probability(test[features])
    ptr = model.attack_probability(train[features])
    thr = model.fit_threshold(yte, p, op["min_precision"], op["fallback_threshold"])
    pred = model.verdict(p)
    _log(f"threshold fitted at {thr:.6f} (>= {op['min_precision']:.0%} precision)")

    # ----- evaluate ---------------------------------------------------------
    metrics = ace.evaluate_flows(yte, p, pred, train_scores=ptr, y_train=ytr)
    per_class = ace.evaluate_attack_classes(test, pred, cls_col)
    gates = opv.evaluate_gates(metrics, mcfg, scfg, per_class)
    _log(f"train ROC={metrics['train_roc']:.4f} | TEST ROC={metrics['roc_auc']:.4f} "
         f"PR={metrics['pr_auc']:.4f} gap={metrics['roc_gap']:.4f}")
    _log(f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
         f"FN%={metrics['fn_rate_%']:.3f}")
    pc = pd.DataFrame(per_class)
    _log("PER-CLASS RECALL:\n" + pc.to_string(index=False))
    pc.to_csv(out_dir / "per_class_metrics.csv", index=False)

    # ----- priority over PREDICTED ALERTS ONLY ------------------------------
    alerts = test.assign(_pred=pred)
    alerts = alerts[alerts._pred == 1].copy()
    priority = PriorityScoreBuilder(pri_cfg["weights"], vol_feature,
                                   pri_cfg["n_deciles"], pri_cfg["log1p_volume"])
    alert_vol = priority.volume_of(alerts)
    alert_p = p[pred == 1]
    priority.fit(alert_p, alert_vol)
    alerts["priority"] = priority.transform_ranked(alert_p, alert_vol)
    alerts["decile"] = pd.qcut(alerts.priority.rank(method="first"),
                               pri_cfg["n_deciles"], labels=False) + 1
    agg = alerts.groupby("decile").agg(mean_priority=("priority", "mean"),
                                       n=("priority", "size"),
                                       hits=(target, "sum")).reset_index()
    agg["hit_%"] = (agg.hits / agg.n * 100).round(2)

    # ----- charts (best-effort) ---------------------------------------------
    charts = {}
    charts["roc_curve"] = pl.roc_curve_plot(yte, p, out_dir / "roc_curve.png",
                                            mcfg["split"]["test_frac"], log=_log)
    charts["pr_curve"] = pl.pr_curve_plot(yte, p, out_dir / "pr_curve.png", log=_log)
    charts["confusion_matrix"] = pl.confusion_matrix_plot(
        metrics, out_dir / "confusion_matrix.png", op["min_precision"], log=_log)
    charts["calibration"] = pl.calibration_plot(yte, p, out_dir / "calibration.png", log=_log)
    charts["decile_priority"] = pl.decile_priority_plot(
        agg, len(alerts), out_dir / "decile_priority.png", pri_cfg["n_deciles"], log=_log)
    gain = model.gain_importance()
    charts["feature_importance"] = pl.feature_importance_plot(
        gain, out_dir / "feature_importance.png", log=_log)
    charts["per_class_recall"] = pl.per_class_recall_plot(
        per_class, out_dir / "per_class_recall.png", log=_log)

    shap_pct = {}
    try:
        xcfg = mcfg["explainability"]
        Xs = test[features].sample(min(xcfg["shap_sample_rows"], len(test)), random_state=seed)
        sv = model.shap_contributions(Xs)
        charts["shap_global"], shap_pct = pl.shap_global_plot(
            sv, features, out_dir / "shap_global.png", log=_log)
        ps = model.attack_probability(Xs)
        i = int(np.argmax(ps))
        charts["shap_local"] = pl.shap_local_plot(sv[i], features, float(ps[i]),
                                                 out_dir / "shap_local.png", log=_log)
    except Exception as exc:                                   # noqa: BLE001
        _log(f"SHAP skipped: {str(exc)[:80]}")

    # ----- EDA over every native column ------------------------------------
    native = ds.native_features(mcfg)
    eda_frame = ds.load_frame(mcfg, columns=native)
    eda = ftp.run(eda_frame, features, native, data.manifest, mcfg, fcfg,
                  out_dir / "eda", log=_log)
    del eda_frame

    # ----- SIEM risk index --------------------------------------------------
    # The risk index is the analyst QUEUE, so it carries the highest-priority
    # ALERTS, capped. Serving-time priority is used (not the in-sample rank) so the
    # exported ranking is the one the API would reproduce for the same flows.
    served_pri = priority.transform(p, priority.volume_of(test))
    is_alert = pred == 1
    queue = (pd.DataFrame({"probability": p[is_alert],
                           "verdict": "ALERT",
                           "priority": served_pri[is_alert],
                           "decile": priority.decile(served_pri[is_alert])})
             .sort_values("priority", ascending=False))
    cap = int(scfg["siem"].get("risk_index_max_records", len(queue)))
    exported = queue.head(cap)
    records = sc.to_risk_records(exported, scfg, seed=seed)
    sc.write_risk_index(records, str(out_dir / "risk_index.jsonl"))
    routing = sc.summarise(records)
    routing["alerts_total"] = int(len(queue))
    routing["exported"] = int(len(exported))
    routing["withheld_by_cap"] = int(len(queue) - len(exported))
    _log(f"risk index: {len(exported):,} of {len(queue):,} alerts exported "
         f"(cap {cap:,}; {len(queue) - len(exported):,} withheld) · {routing}")

    # ----- persist ----------------------------------------------------------
    model.save(str(out_dir / "lgbm_model.pkl"), extra=priority.to_bundle())

    M = {
        "model": "CSE-CIC-IDS2018 Alert-Triage (native-only, LightGBM)",
        "dataset": data.manifest.get("dataset", "CSE-CIC-IDS2018 (NF-v2, NetFlow)"),
        "rows_total": int(len(df)), "rows_train": int(len(tr)), "rows_test": int(len(te)),
        "attack_rate_%": round(float(y.mean() * 100), 4),
        "n_features": len(features), "features": features,
        "protocol": (f"random {round((1 - mcfg['split']['test_frac']) * 100)}/"
                     f"{round(mcfg['split']['test_frac'] * 100)} split "
                     f"({'stratified, ' if mcfg['split']['stratify'] else ''}seed={seed})"),
        "threshold": round(thr, 6), "min_precision_floor": op["min_precision"],
        "scale_pos_weight": round(model.scale_pos_weight, 1),
        "hyperparameters": {k: mcfg["lgbm_classifier"][k] for k in
                            ("n_estimators", "num_leaves", "min_child_samples", "learning_rate")
                            if k in mcfg["lgbm_classifier"]},
        **metrics,
        "per_class": per_class,
        "weakest_attack_class": ace.weakest_class(per_class),
        "decile": agg.to_dict("records"),
        "decile_basis": {"n_alerts": int(len(alerts)), "formula": priority.formula(),
                         "note": "predicted ALERTS only — normal traffic not ranked"},
        "gain_pct": gain.sort_values(ascending=False).round(3).to_dict(),
        "shap_pct": shap_pct,
        "routing": routing,
        "acceptance": gates,
        "eda": {"mean_abs_feature_correlation": eda["mean_abs_feature_correlation"]},
        "charts": [Path(v).name for v in charts.values() if v],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(M, fh, indent=2, default=float)
    with open(out_dir / "triage_scorecard.md", "w", encoding="utf-8") as fh:
        fh.write(_render_scorecard(M))

    _log("=" * 72)
    _log("ACCEPTANCE: ROC %.4f (>=%.2f %s) | PR %.4f (>=%.2f %s) | precision %.3f "
         "(>=%.2f %s)" % (
             metrics["roc_auc"], gates["roc_auc_min"], _pf(gates["roc_pass"]),
             metrics["pr_auc"], gates["pr_auc_min"], _pf(gates["pr_pass"]),
             metrics["precision"], gates["precision_min"], _pf(gates["precision_pass"])))
    _log("            FN %.3f%% (<=%.1f%% %s) | gap %.4f (<=%.3f %s) | reduction %.1f%% "
         "(>=%.0f%% %s)" % (
             metrics["fn_rate_%"], gates["fn_rate_max"] * 100, _pf(gates["fn_pass"]),
             metrics["roc_gap"], gates["roc_gap_max"], _pf(gates["not_overfit_pass"]),
             metrics["alert_reduction_%"], gates["alert_reduction_min"] * 100,
             _pf(gates["reduction_pass"])))
    _log("            FP/analyst-hr %.2f (<=%.1f %s) | weakest class %s recall %.2f "
         "(>=%.2f %s)" % (
             gates["fp_per_analyst_hour"], gates["fp_per_analyst_hour_max"],
             _pf(gates["fp_pass"]), gates["weakest_attack_class"],
             gates["min_recall_any_class"], gates["min_recall_any_class_floor"],
             _pf(gates["class_coverage_pass"])))
    _log("OVERALL: %s" % _pf(gates["all_pass"]))
    _log("Artifacts -> " + str(out_dir))
    return M


def _pct(x):
    try:
        return f"{100 * float(x):.1f}%" if x == x else "N/A"
    except Exception:
        return "N/A"


def _pf(b):
    return "PASS" if b else "FAIL"


def _render_scorecard(m: dict) -> str:
    g = m["acceptance"]
    lines = [
        "# Alert Triage & Prioritization — Triage Scorecard",
        f"**Dataset:** {m['dataset']}  ",
        f"**Protocol:** {m['protocol']}  ",
        f"**Rows:** {m['rows_total']:,} (train/test = {m['rows_train']:,}/{m['rows_test']:,}) "
        f"· attack rate {m['attack_rate_%']}%  ",
        f"**Features:** {m['n_features']} native NetFlow measurements  ",
        f"**Operating point:** {g['operating_point']} -> threshold {m['threshold']}",
        "",
        "## Flow-level performance",
        "| metric | value |", "|---|---|",
        f"| ROC-AUC | {m['roc_auc']} |",
        f"| PR-AUC | {m['pr_auc']} |",
        f"| precision | {m['precision']} |",
        f"| recall | {m['recall']} |",
        f"| FN rate | {m['fn_rate_%']}% |",
        f"| alert reduction | {m['alert_reduction_%']}% |",
        f"| train-vs-test ROC gap | {m['roc_gap']} |",
        "",
        "## Recall per attack class",
        "| attack class | flows | recall |", "|---|---|---|",
        *(f"| {c['attack_class']} | {c['n']:,} | {c['recall']:.4f} |" for c in m["per_class"]),
        "",
        "## Acceptance gates (plan §8)",
        f"- ROC-AUC ≥ {g['roc_auc_min']}: **{_pf(g['roc_pass'])}** ({m['roc_auc']})",
        f"- PR-AUC ≥ {g['pr_auc_min']}: **{_pf(g['pr_pass'])}** ({m['pr_auc']})",
        f"- FN rate ≤ {_pct(g['fn_rate_max'])}: **{_pf(g['fn_pass'])}** ({_pct(g['fn_rate'])})",
        f"- precision ≥ {_pct(g['precision_min'])}: **{_pf(g['precision_pass'])}** "
        f"({_pct(g['precision'])})",
        f"- ROC gap ≤ {g['roc_gap_max']}: **{_pf(g['not_overfit_pass'])}** ({g['roc_gap']})",
        f"- alert reduction ≥ {_pct(g['alert_reduction_min'])}: "
        f"**{_pf(g['reduction_pass'])}** ({_pct(g['alert_reduction'])})",
        f"- FP/analyst-hour ≤ {g['fp_per_analyst_hour_max']}: **{_pf(g['fp_pass'])}** "
        f"({g['fp_per_analyst_hour']:.2f})",
        f"- weakest class recall ≥ {g['min_recall_any_class_floor']}: "
        f"**{_pf(g['class_coverage_pass'])}** ({g['weakest_attack_class']} "
        f"{g['min_recall_any_class']:.2f})",
        f"- **OVERALL: {_pf(g['all_pass'])}**",
        "",
        "## Top drivers (split gain)",
        *(f"- `{k}`  ({v:.1f}%)" for k, v in list(m["gain_pct"].items())[:10]),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_training(skip_screen="--skip-screen" in sys.argv)
