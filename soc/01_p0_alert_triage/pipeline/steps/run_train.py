#!/usr/bin/env python3
"""Train step — runs the model's own training, then maps metrics to gate names.

COPY THIS TO THE MODEL REPO at:
    Alert-Triage-Prioritization/soc/01_p0_alert_triage/pipeline/steps/run_train.py

CONTRACT
  in   $ML_INPUT_DIR/training_data.parquet     verified bytes
       $ML_INPUT_DIR/selected/native_selected.json   from the eda step
       $ML_PARAMS_FILE
  out  $ML_OUTPUT_DIR/lgbm_model.pkl
       $ML_OUTPUT_DIR/metrics.json             gate vocabulary
       $ML_OUTPUT_DIR/per_class_metrics.csv
       $ML_OUTPUT_DIR/triage_scorecard.md

THE METRIC MAPPING, AND WHY SOME KEYS ARE ABSENT
The model emits `recall`, `brier`, `pr_auc`, `roc_auc`. The platform gates read
`recall_malicious`, `brier_score`, `pr_auc`, plus six the model does not
measure: overall_psi, p1_p2_missed, high_critical_suppressed, replay_passed,
holdout_malicious, replay_agent_worthy.

Those six are OMITTED, never defaulted. The gate substitutes a blocking value
for every absent metric on purpose (bhairava-mlops-gates/…/promotion.py:86-105),
so omission blocks promotion — which is the correct outcome for a metric nobody
measured. Writing `overall_psi: 0.0` to make the pipeline look green would be a
fabricated drift verdict and would turn a fail-closed gate into a fail-open one.

Until this model measures those six, it promotes under `gate_profile:
model_acceptance` (its own §8 floors), not the full triage gate.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

SLOT = Path(__file__).resolve().parents[2]
if str(SLOT) not in sys.path:
    sys.path.insert(0, str(SLOT))

from src.config import load_all_configs                         # noqa: E402
from src import data_source as ds                               # noqa: E402
import src.pipeline as P                                        # noqa: E402


def _params() -> dict:
    raw = os.environ.get("ML_PARAMS_FILE")
    if raw and Path(raw).exists():
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    return {}


def stage_inputs(mcfg: dict, input_dir: Path) -> None:
    """Put the verified Parquet and the eda step's feature list where the model
    expects them. The model's own paths are unchanged."""
    supplied = input_dir / "training_data.parquet"
    if not supplied.exists():
        raise SystemExit(f"missing required input: {supplied}")
    target = ds.dataset_path(mcfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != supplied.resolve():
        shutil.copy2(supplied, target)

    # The eda step's declared output, delivered by the platform under the
    # DECLARED contract name (native_selected.json) — profile-independent.
    # The profile-specific names are accepted too for by-hand runs.
    candidates = (
        input_dir / "selected" / "native_selected.json",
        input_dir / "selected" / mcfg["data"]["selected"],
        input_dir / mcfg["data"]["selected"],
        input_dir / "native_selected.json",
    )
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        raise SystemExit(
            "missing required input from the eda step: native_selected.json\n"
            "train declares `needs: [eda]`, so the platform must have "
            "materialised it. Running by hand? Run run_eda.py first."
        )
    shutil.copy2(selected, ds.data_dir(mcfg) / mcfg["data"]["selected"])


def _finite(value) -> float | None:
    """A metric that is not a finite number was not measured.

    NaN and Infinity are dropped rather than coerced. JSON has no literal for
    either, so they reach Postgres as a bare NaN inside a jsonb cast and abort
    the write — a run that trained successfully, answered 503, and left its row
    stuck forever. Dropping is also the honest reading.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def to_gate_metrics(raw: dict) -> dict:
    """Map the model's metrics.json onto the platform's gate vocabulary.

    Only keys we can genuinely measure are emitted. Absent stays absent.
    """
    mapping = {
        "recall_malicious": raw.get("recall"),
        "pr_auc":           raw.get("pr_auc"),
        "brier_score":      raw.get("brier"),
        "roc_auc":          raw.get("roc_auc"),
        "precision":        raw.get("precision"),
        "f1":               raw.get("f1"),
        "mcc":              raw.get("mcc"),
        "log_loss":         raw.get("log_loss"),
        "roc_gap":          raw.get("roc_gap"),
        "threshold":        raw.get("threshold"),
        "samples":          raw.get("rows_total"),
        "rows_train":       raw.get("rows_train"),
        "rows_test":        raw.get("rows_test"),
        "n_features":       raw.get("n_features"),
        # holdout_malicious IS measurable here: true positives + false negatives
        # in the test split is exactly "how many malicious rows the holdout had".
        "holdout_malicious": (raw.get("tp") or 0) + (raw.get("fn") or 0),
    }
    out: dict = {}
    for key, value in mapping.items():
        finite = _finite(value)
        if finite is not None:
            out[key] = finite

    # Percentages the model reports as 0-100; the gates read fractions.
    if _finite(raw.get("fn_rate_%")) is not None:
        out["fn_rate"] = float(raw["fn_rate_%"]) / 100.0
    if _finite(raw.get("alert_reduction_%")) is not None:
        out["alert_reduction"] = float(raw["alert_reduction_%"]) / 100.0

    # Non-numeric context belongs in params, not metrics — a boolean logged as
    # 1.0 can be averaged and charted as though it were a measurement.
    acceptance = raw.get("acceptance") or {}
    if isinstance(acceptance, dict) and "all_pass" in acceptance:
        out["acceptance_all_pass"] = bool(acceptance["all_pass"])

    weakest = raw.get("weakest_attack_class")
    if isinstance(weakest, dict):
        recall = _finite(weakest.get("recall"))
        if recall is not None:
            out["min_recall_any_class"] = recall
    return out


def main() -> int:
    params = _params()
    input_dir = Path(os.environ.get("ML_INPUT_DIR") or (SLOT / "data"))
    out_dir = Path(os.environ.get("ML_OUTPUT_DIR") or (SLOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_all_configs()
    mcfg, fcfg = cfg["model"], cfg["feature"]
    # Dataset profile: platform job parameters win, env is the manual override,
    # cicids2017 is the default (must match the eda step's selection).
    profile = str(params.get("dataset")
                  or os.environ.get("ALERT_TRIAGE_DATASET", "cicids2017")).strip()
    if profile not in ("", "default", "ids2018"):
        P.apply_dataset_profile(mcfg, fcfg, profile,
                                log=lambda m: print(f"[train] {m}", flush=True))
    if "seed" in params:
        mcfg["seed"] = int(params["seed"])
    if "min_precision" in params:
        mcfg["operating_point"]["min_precision"] = float(params["min_precision"])

    stage_inputs(mcfg, input_dir)
    print(f"[train] inputs staged; training into {out_dir}", flush=True)

    # The model's own training, reusing the eda step's screen. Nothing about the
    # model's logic is changed here. run_training() reloads the configs
    # internally, so the profile must travel WITH the call — the overlay applied
    # to this wrapper's own copies does not reach it.
    P.run_training(out_dir=str(out_dir), skip_screen=True,
                   dataset=profile if profile not in ("", "default", "ids2018")
                   else None)

    raw_path = out_dir / "metrics.json"
    if not raw_path.exists():
        raise SystemExit(
            "training finished but wrote no metrics.json — refusing to report "
            "success for a run with no evidence")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    gate_metrics = to_gate_metrics(raw)
    # Keep the model's full output for humans, alongside the gate view.
    (out_dir / "metrics_model_native.json").write_text(
        json.dumps(raw, indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(gate_metrics, indent=2), encoding="utf-8")

    print(f"[train] metrics.json: {len(gate_metrics)} measured keys", flush=True)
    for key in ("recall_malicious", "pr_auc", "brier_score", "roc_auc"):
        if key in gate_metrics:
            print(f"[train]   {key} = {gate_metrics[key]}", flush=True)
    missing = [k for k in ("overall_psi", "p1_p2_missed",
                           "high_critical_suppressed", "replay_passed",
                           "replay_agent_worthy") if k not in gate_metrics]
    if missing:
        print(f"[train] NOT MEASURED (omitted, will block the full triage "
              f"gate): {', '.join(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
