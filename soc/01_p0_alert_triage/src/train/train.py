"""
train.py  —  platform training entrypoint for Alert Triage & Prioritization (#01).

Mirrors the UEBA slot's src/train/train.py role (soc/05_p0_ueba_insider_threat):
a self-contained, environment-variable-driven training entrypoint that the
ml-platform domain scanner discovers at soc/<slot>/src/train/train.py, runs the
model's own verified training, and logs the run + artifacts to MLflow.

It does NOT re-implement the model. The verified end-to-end chain lives in
src/pipeline.py:run_training() (screen -> ingest -> split -> LightGBM -> fitted
threshold -> priority -> evaluation -> §8 gates -> persist). This wrapper only:
  1. resolves the training data (local-first, S3 fallback),
  2. delegates to run_training(), then
  3. logs params/metrics/artifacts to MLflow (best-effort) and prints a
     machine-readable JSON summary to stdout as the job result.

RUN
    python src/train/train.py

ENVIRONMENT (all optional; sensible local defaults)
    ML_PLATFORM_WORKDIR      scratch root (default /tmp/ml-platform-data); used for
                             MPLCONFIGDIR so a headless run does not warn.
    ML_OUTPUT_DIR            where artifacts are written (default <slot>/outputs).
    ML_INPUT_DIR             platform-staged inputs; training_data.parquet is used
                             if present and the local dataset is absent.
    ML_TRAINING_DATA_BUCKET  S3 bucket for the fallback download
                             (default bhairava-ml-training-data).
    MLFLOW_EXPERIMENT_NAME   MLflow experiment (default alert_triage).
    ALERT_TRIAGE_SKIP_SCREEN set to 1 to reuse the committed screened feature set
                             instead of re-running the leak/noise screen.
    ALERT_TRIAGE_DATASET     dataset profile from model_config.dataset_profiles.
                             Defaults to `cicids2017` — the platform's governed
                             table IS the CICIDS2017 (MachineLearningCSV) build,
                             so the platform trains that benchmark: EDA + leak/
                             noise screen keeping the TOP 18 native features,
                             then the supervised LightGBM classifier. Set to
                             `default` for the CSE-CIC-IDS2018 table instead.
"""
import json
import os
import shutil
import sys
from pathlib import Path

# The slot root (<slot>/src/train/train.py -> two parents up is <slot>/src, three
# is the slot). src.pipeline inserts the slot root on sys.path when run directly,
# but we import it as a module here, so make `src` importable first.
SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SLOT_ROOT))

# MPLCONFIGDIR must be set BEFORE matplotlib is imported (src.pipeline imports the
# plotting module at import time), otherwise matplotlib warns it cannot use its
# default config dir. Force the headless Agg backend for the same reason.
WORK = Path(os.environ.get("ML_PLATFORM_WORKDIR", "/tmp/ml-platform-data")) / "alert_triage"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")

from src.config import load_all_configs, resolve_path        # noqa: E402
from src import data_source as ds                            # noqa: E402


MODEL_KEY = "soc/01_p0_alert_triage"
BUCKET = os.environ.get("ML_TRAINING_DATA_BUCKET", "bhairava-ml-training-data")
S3_PREFIX = f"{MODEL_KEY}/processed"
MODEL_NAME = "bhairava.default.alert_triage"


def ensure_dataset(mcfg: dict) -> Path:
    """Resolve the training parquet, local-first with an S3 fallback.

    Order, most-preferred first:
      1. the dataset already present at the slot's data path (the offline/default
         case — data/build_ids2018_dataset.py or a copied-in table),
      2. a platform-staged $ML_INPUT_DIR/training_data.parquet (copied into place),
      3. s3://$ML_TRAINING_DATA_BUCKET/<MODEL_KEY>/processed/<dataset>, only if
         boto3 is importable (it is an optional dep — the S3 path is not needed to
         train from local data).
    """
    target = ds.dataset_path(mcfg)
    if target.exists():
        print(f"[data] using local dataset {target}", flush=True)
        return target

    input_dir = os.environ.get("ML_INPUT_DIR")
    if input_dir:
        staged = Path(input_dir) / "training_data.parquet"
        if staged.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, target)
            print(f"[data] staged platform input {staged} -> {target}", flush=True)
            return target

    try:
        import boto3
    except ImportError as exc:
        raise SystemExit(
            f"training data not found at {target} and no S3 fallback available "
            f"({exc}). Rebuild it with `python data/build_ids2018_dataset.py`, or "
            f"install boto3 and set ML_TRAINING_DATA_BUCKET for the S3 download."
        )

    key = f"{S3_PREFIX}/{mcfg['data']['dataset']}"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading s3://{BUCKET}/{key} -> {target}", flush=True)
    boto3.client("s3").download_file(BUCKET, key, str(target))
    return target


def main() -> int:
    skip_screen = os.environ.get("ALERT_TRIAGE_SKIP_SCREEN", "") not in ("", "0", "false", "False")
    profile = os.environ.get("ALERT_TRIAGE_DATASET", "cicids2017").strip()
    if profile in ("", "default", "ids2018"):
        profile = None

    cfg = load_all_configs()
    mcfg, fcfg = cfg["model"], cfg["feature"]

    # The profile decides WHICH parquet the slot trains on (and its own precision
    # floor / screen policy), so it must be applied BEFORE the dataset is
    # resolved — otherwise ensure_dataset would fetch the wrong table.
    import src.pipeline as P

    profile_out = P.apply_dataset_profile(mcfg, fcfg, profile,
                                          log=lambda m: print(f"[train] {m}", flush=True))
    out_dir = Path(os.environ.get("ML_OUTPUT_DIR") or profile_out or resolve_path("outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(mcfg)

    print(f"[train] training into {out_dir} (profile={profile or 'default'}, "
          f"skip_screen={skip_screen})", flush=True)
    # The profile is already applied to this mcfg/fcfg; run_training re-loads its
    # own config copies, so pass the profile through rather than the patched dicts.
    M = P.run_training(out_dir=str(out_dir), skip_screen=skip_screen, dataset=profile)

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            "training finished but wrote no metrics.json — refusing to report "
            "success for a run with no evidence"
        )

    acceptance = M.get("acceptance") or {}
    numeric = {
        k: float(v)
        for k, v in M.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    summary = {
        "status": "succeeded",
        "metrics": {
            k: M.get(k)
            for k in ("roc_auc", "pr_auc", "precision", "recall", "fn_rate_%",
                      "alert_reduction_%", "roc_gap", "threshold")
            if k in M
        },
        "acceptance_all_pass": bool(acceptance.get("all_pass")),
        "model_name": MODEL_NAME,
        "model_uri": str(out_dir / "lgbm_model.pkl"),
        "parameters": {
            "model_key": MODEL_KEY,
            "seed": mcfg["seed"],
            "n_features": M.get("n_features"),
            "rows_total": M.get("rows_total"),
            "threshold": M.get("threshold"),
        },
    }

    # Log the run + artifacts to MLflow. Best-effort: a tracking outage must not
    # fail an otherwise-successful training job (same policy as the UEBA slot).
    try:
        # MLflow 3.x refuses its local file store unless the caller opts in; keep
        # the offline/default run logging without requiring a tracking server.
        # A real deployment overrides MLFLOW_TRACKING_URI to the registry.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "alert_triage"))
        with mlflow.start_run(run_name="alert_triage_train") as run:
            mlflow.log_params(summary["parameters"])
            mlflow.log_metric("acceptance_all_pass", float(summary["acceptance_all_pass"]))
            mlflow.log_metrics({k.replace("%", "pct").strip("_"): v for k, v in numeric.items()})

            for name in ("lgbm_model.pkl",):
                p = out_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="model")
            for name in ("metrics.json", "triage_scorecard.md", "per_class_metrics.csv"):
                p = out_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="reports")
            for png in sorted(out_dir.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="figures")

            summary["mlflow_run_id"] = run.info.run_id
            summary["model_uri"] = f"runs:/{run.info.run_id}/model/lgbm_model.pkl"
    except Exception as exc:                                    # noqa: BLE001
        print(f"MLflow logging failed: {exc}", flush=True)

    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
