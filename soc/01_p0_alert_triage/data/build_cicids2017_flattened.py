"""
build_cicids2017_flattened.py — STAGE 2 (EDA -> FLATTENED).

Profiles every column of data/cicids2017_raw.parquet, writes the EDA artefacts, and
uses those findings to produce the model-ready table. The flattened table is a
consequence of the EDA, not a parallel guess at it: the same profile drives the
column drops, the inf repair and the provenance audit.

Outputs:
  data/cicids2017_flattened.parquet             X + attack_class + y_is_attack
  data/cicids2017_flattened_manifest.json       feature list, label map, counts
  data/cicids2017_feature_provenance.csv        every RAW column -> kept/dropped + why
  outputs_cicids2017/eda/raw_column_profile.csv the full per-column profile
  outputs_cicids2017/eda/raw_class_counts.png
  outputs_cicids2017/eda/raw_data_quality.png
  outputs_cicids2017/eda/raw_attack_rate_by_day.png
  outputs_cicids2017/eda/raw_eda_summary.json

Run:  py data/build_cicids2017_flattened.py
Then: py src/pipeline.py --dataset cicids2017
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_SLOT_ROOT = Path(__file__).resolve().parents[1]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.evaluation import raw_dataset_profiler as rp   # noqa: E402
from src.features import raw_flattener as rf            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
RAW = HERE / "cicids2017_raw.parquet"
RAW_MANIFEST = HERE / "cicids2017_raw_manifest.json"
FLAT = HERE / "cicids2017_flattened.parquet"
FLAT_MANIFEST = HERE / "cicids2017_flattened_manifest.json"
PROVENANCE = HERE / "cicids2017_feature_provenance.csv"
EDA = _SLOT_ROOT / "outputs_cicids2017" / "eda"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"{RAW.name} missing — run data/build_cicids2017_raw.py first")
    EDA.mkdir(parents=True, exist_ok=True)

    with open(RAW_MANIFEST, encoding="utf-8") as fh:
        raw_manifest = json.load(fh)
    raw_columns = raw_manifest["raw_columns"]
    label_col = raw_manifest["label_column"]
    day_col = raw_manifest["day_column"]

    print(f"RAW: {raw_manifest['rows']:,} rows x {len(raw_columns)} columns")

    # ---- 1) EDA: profile every raw column, one at a time -----------------------
    print("\n1) profiling raw columns")
    profile = rp.profile_columns(RAW, raw_columns, log=print)
    profile.to_csv(EDA / "raw_column_profile.csv", index=False)
    summary = rp.summarise(profile)
    print(f"   decisions: {summary['decisions']}")
    for col, why in summary["dropped"].items():
        print(f"   DROP {col:32s} {why}")
    if summary["inf_repaired"]:
        print(f"   inf to repair in: {summary['inf_repaired']}")

    # ---- 2) EDA charts ---------------------------------------------------------
    print("\n2) EDA charts")
    charts = {}
    charts["raw_class_counts"] = rp.chart_class_counts(
        raw_manifest["label_counts"], EDA / "raw_class_counts.png")
    charts["raw_data_quality"] = rp.chart_data_quality(
        profile, EDA / "raw_data_quality.png")

    # attack rate per capture day, straight off the raw parquet (2 columns only)
    day_lab = pq.read_table(RAW, columns=[day_col, label_col]).to_pandas()
    day_lab[label_col] = rp.normalise_labels(day_lab[label_col])
    day_attack = (day_lab.assign(atk=(day_lab[label_col].str.upper() != "BENIGN"))
                  .groupby(day_col).agg(rows=("atk", "size"), attack_rate=("atk", "mean"))
                  .reset_index())
    charts["raw_attack_rate_by_day"] = rp.chart_attack_rate_by_day(
        day_attack, EDA / "raw_attack_rate_by_day.png")
    print(f"   wrote {sum(1 for v in charts.values() if v)} charts")
    del day_lab

    # ---- 3) flatten using exactly those decisions ------------------------------
    print("\n3) flattening")
    keep = rp.kept_columns(profile)
    raw = pq.read_table(RAW, columns=keep + [label_col]).to_pandas()
    flat, stats = rf.flatten(raw, profile, label_col, log=print)
    del raw

    flat.to_parquet(FLAT, index=False)
    prov = rf.provenance(profile, stats["features"])
    prov.to_csv(PROVENANCE, index=False)
    balance = rf.class_balance(flat)

    attack_rate = float(flat[rf.TARGET].mean())
    manifest = {
        "stage": "flattened",
        "dataset": "CICIDS2017 (CIC MachineLearningCVE, 8 days) — flattened",
        "built_from": RAW.name,
        "rows": int(stats["rows"]),
        "rows_before_dedupe": int(stats["rows_before_dedupe"]),
        "duplicate_rows_removed": int(stats["duplicate_rows_removed"]),
        "native_features": stats["features"],
        "n_native": len(stats["features"]),
        "target": rf.TARGET,
        "attack_class_col": rf.ATTACK_CLASS,
        "label_map": "BENIGN=0, any attack=1",
        "attack_rate_%": round(attack_rate * 100, 4),
        "attack_classes": balance.set_index("attack_class").rows.to_dict(),
        "dropped_raw_columns": summary["dropped"],
        "inf_repaired_columns": stats["inf_repaired_columns"],
        "nan_filled": int(stats["nan_filled"]),
        "provenance": PROVENANCE.name,
    }
    with open(FLAT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    eda_summary = {"raw": {k: raw_manifest[k] for k in
                           ("rows", "n_raw_columns", "attack_rate_%", "n_classes",
                            "rows_per_day", "label_counts")},
                   "column_decisions": summary,
                   "attack_rate_by_day": day_attack.to_dict("records"),
                   "flattened": {k: manifest[k] for k in
                                 ("rows", "rows_before_dedupe", "duplicate_rows_removed",
                                  "n_native", "attack_rate_%", "attack_classes")},
                   "charts": [Path(v).name for v in charts.values() if v]}
    with open(EDA / "raw_eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(eda_summary, fh, indent=2, default=str)

    print(f"\nFLATTENED: {manifest['rows']:,} rows x {manifest['n_native']} features "
          f"({manifest['attack_rate_%']}% attack, {len(manifest['attack_classes'])} classes)")
    print(f"  raw columns dropped: {len(summary['dropped'])} of {len(raw_columns)}")
    print(f"  duplicate rows removed: {manifest['duplicate_rows_removed']:,}")
    print("\nCLASS BALANCE (after dedupe):")
    print(balance.to_string(index=False))
    print(f"\nwrote data/{FLAT.name} ({FLAT.stat().st_size / 1024 / 1024:,.1f} MB)")
    print(f"wrote data/{FLAT_MANIFEST.name}")
    print(f"wrote data/{PROVENANCE.name}")
    print(f"wrote outputs_cicids2017/eda/ (profile + {len(eda_summary['charts'])} charts + summary)")
    print("\nnext: py src/pipeline.py --dataset cicids2017")


if __name__ == "__main__":
    main()
