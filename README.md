# Model #01 — Alert Triage & Prioritization (CSE-CIC-IDS2018, LightGBM)

A SOC alert-triage model: LightGBM over screened native NetFlow features, scoring
and ranking flows so analysts work the highest-risk traffic first. Laid out in the
project's standard model-slot structure — `configs/` + `mlops/` + `tests/` +
`src/{features,models,evaluation,serving}`.

```
requirements.txt                                  repo-level floors
docs/model_card_registry/                         registry pointer
soc/01_p0_alert_triage/CICIDS_lightgbm/
  MODEL_NOTES.md        <- read this first: what moved where, and why
  requirements.txt      <- the pinned, verified deps (was missing upstream)
  configs/              model_config · feature_config · serving_config
  data/                 merged NetFlow parquet + manifests + builders
  mlops/                model_card · drift_config · retrain_schedule
  notebooks/            01_eda_cicids2017.ipynb
  outputs/              lgbm_model.pkl · metrics.json · 12 charts · deck
  postman/              collection with real per-attack-class flows
  src/                  config · data_source · pipeline
                        features/ models/ evaluation/ reporting/ serving/
  tests/                47 tests — acceptance gates, screen, API, configs
```

## Quick start

```bash
cd soc/01_p0_alert_triage/CICIDS_lightgbm
pip install -r requirements.txt

py src/pipeline.py          # full build: screen → train → eda → reports
py -m pytest -q             # 47 tests
py src/serving/api.py       # FastAPI on 127.0.0.1:8081
```

## Verified build — 2026-08-12, Python 3.14.2

`py src/pipeline.py` ran all 7 steps green in ~78s and reproduced the reference run
**exactly** — every metric in `outputs/metrics.json` matches, and
`data/native_selected.json` and `data/feature_screen.csv` are byte-identical.

| | |
| --- | --- |
| ROC-AUC | 0.9922 (train 0.9944, gap 0.0022) |
| PR-AUC | 0.9864 |
| Operating point | threshold 0.144544 — highest recall at ≥70% precision |
| Precision / recall | 0.7003 / 0.9821 |
| FN rate | 1.79% (889 missed of 49,597 attacks) |
| Alert reduction | 68.6% |
| Features | 5 native NetFlow measurements |
| Rows | 738,607 (22.38% attack, 14 attack classes) |

`py -m pytest -q` → **47 passed**. API verified live: a real *DoS attacks-Hulk*
flow scores 0.999992 → `ALERT`, decile 10, action `escalate`; a *Benign* flow
scores 0.004291 → `NORMAL`, routed to `data_lake`.

This is a **public-benchmark** result, not a production claim — see
`mlops/model_card.yaml` limitations before drawing conclusions about live traffic.
