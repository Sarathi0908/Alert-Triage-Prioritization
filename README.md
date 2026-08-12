# Alert Triage & Prioritization — CSE-CIC-IDS2018 · LightGBM

A SOC alert-triage model: LightGBM over screened native NetFlow features, scoring
and ranking flows so analysts work the highest-risk traffic first. The model emits a
probability, a graded priority and an ALERT/NORMAL verdict; the backend routes NORMAL
to the Data Lake and ALERT to the analyst priority queue.

Laid out in the project's standard model-slot structure — the same one the UEBA
(insider-threat) slot uses.

```
soc/01_p0_alert_triage/
  configs/      feature_config · model_config · serving_config
  data/         dataset builders · manifests · screened feature set · screen audit
  mlops/        model_card · drift_config · retrain_schedule
  notebooks/    01..05 numbered series + build_notebooks.py
  outputs/      bundle · metrics · scorecard · 12 charts · risk index   (gitignored)
  postman/      collection built from real per-attack-class flows
  src/          config · data_source · pipeline
                features/ models/ evaluation/ serving/
  tests/        129 tests, one module per source module
  make_execution_deck.py · make_feature_sheets.py · make_postman_collection.py
```

`soc/01_p0_alert_triage/MODEL_NOTES.md` is the file to read first: it documents the
structure, the naming convention behind every filename, and what changed from the
original flat layout.

## Quick start

```bash
cd soc/01_p0_alert_triage
pip install -r requirements.txt

py src/pipeline.py                 # screen -> train -> evaluate -> EDA -> artefacts
py -m pytest -q                    # 129 tests
py src/serving/flow_triage_api.py  # FastAPI on 127.0.0.1:8081
```

The datasets are not committed — they are large and regenerable. Rebuild with
`py data/build_ids2018_dataset.py` (set `BHAIRAVA_DOWNLOADS` if the benchmark
archive is not in `~/Downloads`).

## Verified — 2026-08-12, Python 3.14.2

`py src/pipeline.py` ran green in ~66s; every §8 acceptance gate passes.

| | |
| --- | --- |
| ROC-AUC | 0.9922 (train 0.9944, gap 0.0022) |
| PR-AUC | 0.9864 |
| Operating point | threshold 0.144544 — **fitted** as the highest recall holding ≥70% precision |
| Precision / recall | 0.7003 / 0.9821 |
| FN rate | 1.79% (889 missed of 49,597 attacks) |
| Alert reduction | 68.6% of flows suppressed as NORMAL |
| FP / analyst-hour | 3.60 |
| Features | 5 native NetFlow measurements (leak/noise/dominance screened) |
| Rows | 738,607 · 22.38% attack · 14 attack classes |

`py -m pytest -q` → **129 passed**. API verified live: a real *DoS attacks-Hulk* flow
scores 0.999992 → ALERT, priority 86.5, decile 10, `escalate`; a *Benign* flow scores
0.004291 → NORMAL, routed to `data_lake`.

Recall is 1.00 for every attack class except **Infilteration** at 0.70 — that is the
model's blind spot, and the reason per-class recall carries its own acceptance gate.

This is a **public-benchmark** result, not a production claim. The capture is
synthetic; re-validate on real flow telemetry in shadow mode and re-fit the threshold
to analyst capacity before any automation. See
`soc/01_p0_alert_triage/mlops/model_card.yaml` for the full limitations.
