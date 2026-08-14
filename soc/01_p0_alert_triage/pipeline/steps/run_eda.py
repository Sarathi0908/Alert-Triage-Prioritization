#!/usr/bin/env python3
"""EDA step — screen the native features and write the drift baseline.

COPY THIS TO THE MODEL REPO at:
    Alert-Triage-Prioritization/soc/01_p0_alert_triage/pipeline/steps/run_eda.py

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
The model already contains everything needed here — `run_screen()` at
src/pipeline.py:63 and the flow profiler at src/evaluation/flow_traffic_profiler.py.
This wrapper does not reimplement any of it. It only:

  1. puts the platform-supplied Parquet where the model's config expects it,
  2. calls the model's own screen,
  3. adds the ONE artifact the model does not produce today — a training-time
     drift baseline — and
  4. copies the results to $ML_OUTPUT_DIR so the platform can keep them.

The model's own logic is untouched. That is the point: the contract adapts to
the model, not the other way round.

CONTRACT
  in   $ML_INPUT_DIR/training_data.parquet   already sha256-verified
       $ML_PARAMS_FILE                       resolved params as JSON
  out  $ML_OUTPUT_DIR/native_selected.json
       $ML_OUTPUT_DIR/feature_screen.csv
       $ML_OUTPUT_DIR/eda/…
       $ML_OUTPUT_DIR/eda/drift_baseline.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SLOT = Path(__file__).resolve().parents[2]      # …/soc/01_p0_alert_triage
if str(SLOT) not in sys.path:
    sys.path.insert(0, str(SLOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from src.config import load_all_configs                         # noqa: E402
from src import data_source as ds                               # noqa: E402
import src.pipeline as P                                        # noqa: E402


def _env_dir(name: str, default: str) -> Path:
    path = Path(os.environ.get(name) or (SLOT / default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _params() -> dict:
    raw = os.environ.get("ML_PARAMS_FILE")
    if raw and Path(raw).exists():
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    return {}


def stage_input(mcfg: dict, input_dir: Path) -> Path:
    """Place the platform's Parquet where model_config.yaml says the data lives.

    Nothing in the model is modified — `data.dataset` still names the file it
    always named. We simply make that filename BE the bytes the platform
    verified, so the model reads platform-governed data without knowing it.

    When the platform did NOT stage an input (older executors set no
    ML_INPUT_DIR contract), fall back to the training entrypoint's own
    resolution: local table first, then the S3 download — the UEBA mirror.
    """
    supplied = input_dir / "training_data.parquet"
    if not supplied.exists():
        from src.train.train import ensure_dataset
        return ensure_dataset(mcfg)
    target = ds.dataset_path(mcfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != supplied.resolve():
        shutil.copy2(supplied, target)
    return target


def build_drift_baseline(frame: pd.DataFrame, features: list[str],
                         target: str) -> dict:
    """Decile bin edges + proportions per feature — the PSI reference.

    This is the artifact that makes drift detection answer the right question.
    The platform's monitor currently computes PSI against a rolling window
    (monitoring/monitor/daily.py:722-723), i.e. "this week vs last week", so a
    model can drift steadily away from its own training distribution and never
    register drift. A baseline captured HERE, at training time, is what the
    comparison should be against.
    """
    baseline: dict = {
        "schema_version": 1,
        "n_rows": int(len(frame)),
        "target": target,
        "label_balance": {
            "positive_rate": float(frame[target].mean()),
            "positive": int(frame[target].sum()),
            "negative": int(len(frame) - frame[target].sum()),
        },
        "features": {},
    }
    for col in features:
        values = pd.to_numeric(frame[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            # Omitted, never zero-filled: a feature we could not profile is
            # not a feature with a flat distribution.
            baseline["features"][col] = {"status": "unmeasured"}
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            baseline["features"][col] = {
                "status": "constant", "value": float(values.iloc[0])}
            continue
        counts, _ = np.histogram(values, bins=edges)
        total = counts.sum() or 1
        baseline["features"][col] = {
            "status": "ok",
            "edges": [float(e) for e in edges],
            "proportions": [float(c / total) for c in counts],
            "null_rate": float(1.0 - len(values) / len(frame)),
        }
    return baseline


def main() -> int:
    params = _params()
    input_dir = Path(os.environ.get("ML_INPUT_DIR") or (SLOT / "data"))
    out_dir = _env_dir("ML_OUTPUT_DIR", "outputs")
    (out_dir / "eda").mkdir(parents=True, exist_ok=True)

    cfg = load_all_configs()
    mcfg, fcfg = cfg["model"], cfg["feature"]
    profile = os.environ.get("ALERT_TRIAGE_DATASET", "cicids2017").strip()
    if profile not in ("", "default", "ids2018"):
        P.apply_dataset_profile(mcfg, fcfg, profile,
                                log=lambda m: print(f"[eda] {m}", flush=True))
    if "seed" in params:
        mcfg["seed"] = int(params["seed"])
    if "screen_sample_rows" in params:
        fcfg.setdefault("screen", {})["sample_rows"] = int(params["screen_sample_rows"])

    staged = stage_input(mcfg, input_dir)
    print(f"[eda] input staged at {staged}", flush=True)

    # ---- the model's own screen, unmodified --------------------------------
    selected = P.run_screen(mcfg, fcfg)
    print(f"[eda] screen selected {len(selected)} features", flush=True)

    # ---- the drift baseline the model does not produce ---------------------
    frame = ds.load_frame(mcfg, columns=selected)
    baseline = build_drift_baseline(frame, selected, mcfg["target"])
    (out_dir / "eda" / "drift_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"[eda] drift baseline over {len(selected)} features "
          f"({baseline['n_rows']:,} rows, "
          f"positive rate {baseline['label_balance']['positive_rate']:.4f})",
          flush=True)

    # ---- publish everything the platform declared as an output -------------
    data_dir = ds.data_dir(mcfg)
    for name in (mcfg["data"]["selected"], mcfg["data"]["screen_audit"]):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # The model writes its EDA charts under outputs/eda when the training step
    # runs the profiler; anything already there is carried forward.
    slot_eda = SLOT / "outputs" / "eda"
    if slot_eda.exists() and slot_eda.resolve() != (out_dir / "eda").resolve():
        for item in slot_eda.iterdir():
            if item.is_file():
                shutil.copy2(item, out_dir / "eda" / item.name)

    print(f"[eda] outputs written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
