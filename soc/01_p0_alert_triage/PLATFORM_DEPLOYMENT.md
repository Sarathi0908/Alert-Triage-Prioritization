# Deploying Alert Triage #01 on the ML Platform (http://3.7.110.130:8090)

The slot mirrors the UEBA slot's platform integration: the worker clones the
repo, pip-installs `requirements.txt`, runs `src/train/train.py`, and the
entrypoint resolves its own training data (platform-staged `$ML_INPUT_DIR`
first, then the `s3://bhairava-ml-training-data` download) and prints a JSON
summary to stdout.

**What the platform trains:** the CICIDS2017 (MachineLearningCSV) benchmark —
EDA + leak/noise screen keeping the **top 18 native features**, then the
supervised LightGBM triage classifier at the 0.95-precision operating point.
(`ALERT_TRIAGE_DATASET=default` switches back to the CSE-CIC-IDS2018 table.)

## Why the 2026-08-13 run failed (tjob_d022146fa49a4b21)

1. **Wrong trigger.** The job was created by hand on the Training page with
   placeholder values: `model_project_key=project.01_p0_alert_triage` (the
   registered key is `project.repo_acfdcaac8f434c68.01_p0_alert_triage`),
   `model_project_version_id=MPV`, `dataset_version_id=v1`, and a
   double-prefixed pipeline key under tenant `bhairava`. The worker failed in
   551 ms with `RuntimeError: model project is not registered` — it never
   cloned the repo. Successful UEBA runs are dispatched from the repository
   **Sync** action, which fills the real keys.
2. **No training data reachable.** `data/*.parquet` is gitignored, so the
   worker's clone has no table; the S3 fallback needed `boto3` (it was
   commented out of requirements) and the bucket had no alert-triage objects.
3. **Wrong dataset profile.** The default config trains CSE-CIC-IDS2018;
   the data this platform governs is CICIDS2017.

All three are fixed in this commit (boto3 pinned, CICIDS2017 is the platform
default profile, top-18 screen policy, pipeline-step handoff repaired).

## One-time install

### 1. Upload the training table to MinIO (from the platform server)

MinIO is not exposed publicly, so copy the parquet to the server first:

```bash
scp soc/01_p0_alert_triage/data/cicids2017_flattened.parquet <user>@3.7.110.130:/tmp/
```

Then on the server (adjust the mc alias/creds to the stack's MinIO):

```bash
mc cp /tmp/cicids2017_flattened.parquet \
  local/bhairava-ml-training-data/soc/01_p0_alert_triage/processed/cicids2017_flattened.parquet
```

(Any equivalent works — `aws s3 cp --endpoint-url http://localhost:9000 …`.
The key must be exactly `soc/01_p0_alert_triage/processed/cicids2017_flattened.parquet`.)

### 2. Push this commit and sync the repository

Push to `feature/01-alert-triage-CICIDS2017` on
`github.com/Sarathi0908/Alert-Triage-Prioritization`, then in the UI:

**Model Projects → repository `Alert-Triage-Prioritization` → Sync.**

Sync re-scans the repo at the new commit and dispatches a training job with
the correct registered keys (this is exactly how the UEBA runs are started).
Do **not** create the job by hand on the Training page.

### 3. Watch the run

Training → the new job → logs. Expected sequence:

- `installing project requirements from requirements.txt`
- `[train] dataset profile 'cicids2017': cicids2017_flattened.parquet · precision floor 0.95`
- `[data] downloading s3://bhairava-ml-training-data/soc/01_p0_alert_triage/processed/cicids2017_flattened.parquet`
- screen output ending in `SELECTED 18 native features`
- gates block ending `OVERALL: PASS`
- a final one-line JSON summary with `"status": "succeeded"`.

The run also logs params/metrics/artifacts to MLflow experiment
`alert_triage` (best-effort).

## Knobs

| env var | default | meaning |
|---|---|---|
| `ALERT_TRIAGE_DATASET` | `cicids2017` | dataset profile (`default` = IDS2018) |
| `ALERT_TRIAGE_SKIP_SCREEN` | off | reuse the committed 18-feature set (`data/cicids2017_native_selected.json`) instead of re-screening — much faster, same features |
| `ML_TRAINING_DATA_BUCKET` | `bhairava-ml-training-data` | S3 fallback bucket |

If the worker is memory-constrained, set `ALERT_TRIAGE_SKIP_SCREEN=1`: the
screen (a 1.5M-row LightGBM ranking fit) is the expensive part, and its
result — the top-18 feature list — is committed and reproducible.

## Known platform-side gaps (not fixable from this repo)

- The **data-refresh** policies the platform bootstrapped fail validation:
  their dataset keys (`dataset.default.01_p0_alert_triage.training_data`)
  contain dots in the name segment, which the refresh worker's
  `dataset.{tenant}.{name}` family regex rejects. The alert-triage data
  source also inherited UEBA's `required_objects`
  (`features_userday.csv`, `labels.csv`). Training does not depend on the
  refresh path — the entrypoint fetches its table directly — but the
  Data Refresh page will keep showing failed jobs until the platform fixes
  key generation.
