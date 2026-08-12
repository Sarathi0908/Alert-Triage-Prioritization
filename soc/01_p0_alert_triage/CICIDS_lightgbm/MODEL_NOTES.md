# Model #01 — Alert Triage & Prioritization (CSE-CIC-IDS2018, LightGBM)

Standalone copy of the `CICIDS_lightgbm` alert-triage model, restructured to the
project's standard model-slot layout — the same one the UEBA (insider-threat) slot
uses: `configs/` + `mlops/` + `tests/` +
`src/{features,models,evaluation,serving}`.

The model itself is unchanged — same LightGBM, same 5 native NetFlow features,
same fitted operating point. What changed is where the code lives, and that every
number it used to hardcode now comes from `configs/`.

---

## 1 · Where each file went

The source tree was flat (`src/build_*.py`). Files were **renamed** to say what
they are rather than what they build, and moved into the subpackage that owns them.

| Source (`CICIDS_lightgbm/…`)   | Here                                    | Why |
| ------------------------------ | --------------------------------------- | --- |
| `src/build_model.py`           | `src/models/lightgbm_triage.py`         | It *is* the model, not a build script |
| `src/select_features.py`       | `src/features/feature_screener.py`      | Feature engineering → `src/features/` |
| `src/build_eda.py`             | `src/evaluation/eda_report.py`          | Evaluation/exploration → `src/evaluation/` |
| `src/build_excels.py`          | `src/reporting/build_excels.py`         | Artefact builders grouped together |
| `src/build_deck.py`            | `src/reporting/build_deck.py`           | ” |
| `src/build_collection.py`      | `src/reporting/build_collection.py`     | ” |
| `src/make_notebook.py`         | `src/reporting/make_notebook.py`        | ” |
| `src/serving/api.py`           | `src/serving/api.py`                    | Unchanged location |
| `data/build_*_dataset.py`      | `data/build_*_dataset.py`               | Unchanged — see *Deviations* |
| *(new)*                        | `src/config.py`                         | Slot-root + YAML config loader |
| *(new)*                        | `src/data_source.py`                    | All parquet/manifest loading |
| *(new)*                        | `src/pipeline.py`                        | One entry point for the whole chain |
| *(new)*                        | `src/evaluation/plots.py`               | Shared chart style (was duplicated) |

Everything else — `data/*.parquet`, `data/*.json`, `data/*.csv`, `data/*.xlsx`,
all of `outputs/` (model bundle, metrics, 12 PNGs, deck), `notebooks/`,
`postman/` — was copied byte-for-byte.

**Path anchors were re-based.** A module at `src/build_model.py` found the slot
root with `parents[1]`; at `src/models/lightgbm_triage.py` that is `parents[2]`.
Every moved module now takes its paths from `src.config` instead of recomputing
them, so this class of bug cannot recur. `src/serving/api.py` already used
`parents[2]` and did not move, so it was left alone.

## 2 · Files that did not exist before

- **`requirements.txt`** — the headline gap. Derived from the actual imports of
  this slot, pinned to versions verified on 2026-08-12 under Python 3.14.2. It is
  **not** a copy of the sibling `lightgbm/requirements.txt`, which is wrong for
  this model in three ways: it lists `flask` (this serves on **FastAPI**), lists
  `shap` (explainability uses LightGBM's own `pred_contrib`, so `shap` is never
  imported), lists `seaborn` (nothing imports it) — and **omits `pyarrow`**, which
  is mandatory here because every entry point reads a parquet table.
- **`configs/model_config.yaml`**, **`feature_config.yaml`**, **`serving_config.yaml`**
  — the three-config convention this slot uses on `HEAD`. Seeds, split, LightGBM
  hyperparameters, the 0.70 precision floor, the 0.70/0.30 priority weights, all
  screen thresholds, the API port and the Postman scenarios were lifted out of the
  code into these.
- **`mlops/model_card.yaml`**, **`drift_config.yaml`**, **`retrain_schedule.yaml`**
- **`tests/`** — `conftest.py` plus four test modules (see §4).
- **`plan/`** — the slot's implementation plan, present locally but **not
  published**: it carries roadmap prioritisation and staffing detail. `.gitignore`
  excludes it so `git add -A` cannot pull it in. References to it in the model card
  and elsewhere appear as "plan §N".
- **`.gitignore`**, **`pytest.ini`**, this file, and the repo-level
  `requirements.txt` + `docs/model_card_registry/` entry.

## 3 · Running it

```bash
pip install -r requirements.txt

py src/pipeline.py                    # screen → train → eda → excels → notebook → postman → deck
py src/pipeline.py --steps train      # just retrain
py src/pipeline.py --skip screen deck # everything else
py src/pipeline.py --list             # show the steps

py -m pytest -q                       # the acceptance gates, as tests
py src/serving/api.py                 # FastAPI on 127.0.0.1:8081
```

Each pipeline step runs in its own interpreter, so a failing report builder cannot
corrupt the run that produced the model.

## 4 · Tests

| File | Covers |
| ---- | ------ |
| `tests/test_acceptance_gate.py` | Plan §8 gates against `outputs/metrics.json`: ROC/PR floors, FN ceiling, the precision floor the threshold is *fitted* to hold, train-vs-test ROC gap, alert-volume reduction, per-class coverage |
| `tests/test_feature_screen.py`  | Screen invariants on `data/native_selected.json`: no leak-grade feature kept, gain balanced between floor and cap, every dropped column carries a reason, audit CSV agrees with the JSON |
| `tests/test_scoring_contract.py`| `src/serving/api.py` via `TestClient`: body-is-the-feature-vector, served threshold equals the evaluated one, unknown fields ignored, normal flows unranked, batch agrees with single |
| `tests/test_config.py`          | Config coherence: priority weights sum to 1, the precision gate cannot exceed the fitted floor, decile counts agree across configs, `drift_config` watches features the model actually has |

Tests that need a trained model **skip** rather than fail when `outputs/` is
empty, so a fresh clone does not report failures for work it has not done.

## 5 · Deliberate deviations from the UEBA layout

1. **`src/reporting/`** is an extra subpackage. UEBA has
   `features/models/evaluation/serving`; this model additionally builds an Excel
   workbook, a PPTX deck, a Postman collection and a notebook, and those four
   belong together rather than loose in `src/`.
2. **`data/build_*_dataset.py` stayed in `data/`.** UEBA ingests via
   `src/data_source.py` and bootstraps through `_infra`. These two are standalone
   dataset builders that read a zip from `~/Downloads` and write their parquet
   next to themselves; moving them into `src/` would have gained nothing and
   broken that locality. `src/data_source.py` still owns all *reading*.
3. **`src/config.py` degrades gracefully without `_infra/`.** The in-repo version
   raises at import time if no `_infra/` directory is found above the slot. This
   tree has none, so `_find_repo_root` returns `None` and `infra()` raises only if
   a shared-platform module is actually requested. `HAS_INFRA` exposes which mode
   you are in. The `_infra/...` paths in `mlops/*.yaml` are therefore aspirational
   here — they resolve when the slot is placed back inside a full checkout.
4. **No `src/models/*` split into multiple model classes.** UEBA has four model
   modules because it composes baselines and a forest; this is one classifier.

## 6 · Verification of this copy — 2026-08-12, Python 3.14.2

`py src/pipeline.py --keep-going` ran all seven steps green in ~78s, and the
restructured code **reproduced the reference run exactly**:

- Every field compared in `outputs/metrics.json` is identical — `roc_auc` 0.9922,
  `pr_auc` 0.9864, `roc_gap` 0.0022, `threshold` 0.144544, precision 0.7003,
  recall 0.9821, `tn/fp/fn/tp` 151140/20846/889/48708, `alert_reduction_%` 68.61.
- `data/native_selected.json` and `data/feature_screen.csv` are **byte-identical**
  to the source tree: same 5 features in the same order, `TCP_WIN_MAX_IN` dropped
  as dominant, 1 near-constant + 7 redundant + 23 low-gain-tail removed, no leak.
- `py -m pytest -q` → **47 passed**.
- The API was booted for real (not just `TestClient`): a genuine *DoS attacks-Hulk*
  flow scores 0.999992 → `ALERT`, priority 86.5, decile 10, action `escalate`; a
  *Benign* flow scores 0.004291 → `NORMAL`, routed to `data_lake`.

One thing to know about that byte-for-byte result. `src/data_source.subsample`
deliberately draws without replacement *even when the sample covers the whole
table*, which returns a permutation rather than the original row order. The first
build short-circuited that case, and the reported gain percentages came out ~0.1pp
off (40.655 vs 40.632 for `L4_DST_PORT`) — the selected features, the removals and
every trained metric were unaffected, but the numbers no longer matched. The cause
is that the *screening* ranker runs without LightGBM's `deterministic` /
`force_row_wise` flags, so its threaded histogram sums are mildly order-sensitive.
The final model does set both flags and is order-invariant.

Two warnings are expected and neither is a defect in this slot:

- `RuntimeWarning: overflow encountered in reduce` during the EDA step. The column
  profiler sums float32 NetFlow byte counters across 738k rows; some totals exceed
  float32 range. Pre-existing upstream behaviour, affects reported `mean` in
  `outputs/eda/all_columns_profile.csv` for the largest counters only — no model
  input passes through it.
- `StarletteDeprecationWarning: Using httpx with starlette.testclient is
  deprecated; install httpx2`. Third-party; `httpx==0.28.1` still works and is
  pinned in `requirements.txt` with a note.

## 7 · Things worth knowing about the model

- **The threshold is fitted, not fixed.** `pick_threshold` takes the
  highest-recall point that still holds ≥70% precision (0.1445 on the reference
  run), and it is stored *inside* `outputs/lgbm_model.pkl`. The API reads it from
  the bundle, so the served operating point cannot drift from the evaluated one.
  `tests/test_scoring_contract.py` asserts exactly that.
- **Priority ranks predicted alerts only.** Normal flows are never ranked, and
  `/score` returns `priority: null` for them. The blend interpolates against
  `p_grid`/`vol_grid` percentile grids captured at training time — also in the
  bundle.
- **Explainability needs no extra dependency.** `booster_.predict(pred_contrib=True)`
  is exact tree SHAP from LightGBM itself.
- **5 features is deliberate.** The dominance screen drops any feature holding
  more than 45% of the gain (a dominator is a leak/memorisation risk) and any
  below 1.5% (dead weight). `TCP_WIN_MAX_IN` was removed as dominant;
  `DURATION_OUT` was hand-excluded at 0.93 solo AUC.
- **`build_model.py` trained on `ids2018_merged.parquet`** despite the CICIDS2017
  filenames and docstrings throughout the source. The 2017 table is still here as
  a second benchmark (`data.alt_dataset`) but nothing trains on it. Config and
  docstrings now say so plainly.
