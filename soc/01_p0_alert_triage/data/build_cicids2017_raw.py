"""
build_cicids2017_raw.py — STAGE 1 (RAW): the CICIDS2017 MachineLearningCVE set,
8 day-CSVs, assembled into ONE parquet with nothing cleaned.

This is deliberately the *raw* table. The only changes are lossless bookkeeping:
  * CIC ships column names with leading spaces ("  Flow Duration") — stripped, or
    nothing downstream can address a column by name.
  * `_day` records which capture day the row came from.
  * measurement columns are pinned to float64 so all 8 days share one parquet
    schema. Lossless at these magnitudes, and inf/NaN survive intact.
No dedupe, no dropna, no inf handling, no label mapping, no column drops. Those
are flattening decisions, and they are made in STAGE 2 from what the EDA finds —
so the EDA can actually see the problems (inf in Flow Bytes/s, the duplicated
Fwd Header Length.1 column, the mojibake in the Web Attack labels).

Memory: 2.8M rows x 79 columns does not need to be resident. Each CSV is read,
written as a parquet row group, and released, so peak usage is one day-file.

Source directory, in priority order:
  1. $CICIDS2017_DIR
  2. $BHAIRAVA_DOWNLOADS/MachineLearningCSV
  3. ~/Downloads/MachineLearningCSV
It is searched recursively for *.csv, so either the zip's own
MachineLearningCVE/ subfolder or a flat extraction works.

Outputs:
  data/cicids2017_raw.parquet
  data/cicids2017_raw_manifest.json

Run:  py data/build_cicids2017_raw.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parent

LABEL = "Label"
DAY_COL = "_day"


def source_dir() -> Path:
    """Resolve the extracted MachineLearningCVE directory without hardcoding a path."""
    candidates = []
    if os.getenv("CICIDS2017_DIR"):
        candidates.append(Path(os.environ["CICIDS2017_DIR"]))
    dl = os.getenv("BHAIRAVA_DOWNLOADS")
    if dl:
        candidates.append(Path(dl) / "MachineLearningCSV")
    candidates.append(Path("~/Downloads/MachineLearningCSV").expanduser())
    for c in candidates:
        c = c.expanduser()
        if c.is_dir() and any(c.rglob("*.csv")):
            return c
    raise SystemExit(
        "CICIDS2017 CSVs not found. Set CICIDS2017_DIR to the extracted "
        "MachineLearningCVE directory. Tried: "
        + ", ".join(str(c) for c in candidates))


def day_of(path: Path) -> str:
    """'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' -> capture-day label."""
    return path.name.split(".pcap")[0]


def main() -> None:
    src = source_dir()
    files = sorted(src.rglob("*.csv"))
    print(f"source: {src}")
    print(f"{len(files)} day-CSVs\n")

    out_parquet = HERE / "cicids2017_raw.parquet"
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    schema_cols: list[str] | None = None
    per_day: dict[str, int] = {}
    labels: dict[str, int] = {}
    total = 0

    try:
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            df[DAY_COL] = day_of(f)

            if schema_cols is None:
                schema_cols = list(df.columns)
            elif list(df.columns) != schema_cols:
                # Reindex rather than fail: a column-order difference between days is
                # harmless, a missing column is not — it would become silent NaN, so
                # report it loudly.
                missing = set(schema_cols) - set(df.columns)
                extra = set(df.columns) - set(schema_cols)
                if missing or extra:
                    print(f"  !! {f.name}: missing={sorted(missing)} extra={sorted(extra)}")
                df = df.reindex(columns=schema_cols)

            # One parquet file needs one schema, but a column that happens to be
            # all-integer on Monday and fractional on Wednesday would arrive as
            # int64 then double. Pin every measurement to float64: lossless for
            # these magnitudes and it preserves the inf/NaN the EDA must see.
            for c in schema_cols:
                if c not in (LABEL, DAY_COL):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[LABEL] = df[LABEL].astype("string")
            df[DAY_COL] = df[DAY_COL].astype("string")

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_parquet, schema, compression="snappy")
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)

            n = len(df)
            total += n
            per_day[day_of(f)] = n
            for k, v in df[LABEL].astype(str).str.strip().value_counts().items():
                labels[k] = labels.get(k, 0) + int(v)
            print(f"  {f.name:60s} {n:>9,} rows")
            del df, table
    finally:
        if writer is not None:
            writer.close()

    feature_cols = [c for c in schema_cols if c not in (LABEL, DAY_COL)]
    attack = sum(v for k, v in labels.items() if k.upper() != "BENIGN")

    manifest = {
        "stage": "raw",
        "dataset": "CICIDS2017 (CIC MachineLearningCVE, 8 days)",
        "source_dir": str(src),
        "rows": total,
        "n_raw_columns": len(feature_cols),
        "raw_columns": feature_cols,
        "label_column": LABEL,
        "day_column": DAY_COL,
        "rows_per_day": per_day,
        "label_counts": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "attack_rows": attack,
        "attack_rate_%": round(attack / total * 100, 4) if total else 0.0,
        "n_classes": len(labels),
        "cleaning_applied": ["strip leading spaces from column names",
                             f"add {DAY_COL}",
                             "measurements pinned to float64 for one parquet schema "
                             "(lossless; inf/NaN preserved)"],
        "cleaning_deferred_to_flatten": [
            "duplicate-column removal (Fwd Header Length.1)",
            "inf/-inf handling (Flow Bytes/s, Flow Packets/s)",
            "constant / all-zero column removal",
            "row dedupe", "numeric coercion",
            "label normalisation (mojibake dash) + y_is_attack mapping"],
    }
    with open(HERE / "cicids2017_raw_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    size_mb = out_parquet.stat().st_size / 1024 / 1024
    print(f"\nTOTAL: {total:,} rows x {len(feature_cols)} raw columns "
          f"({len(labels)} classes, {manifest['attack_rate_%']}% attack)")
    print("\nLABELS:")
    for k, v in manifest["label_counts"].items():
        print(f"  {k:32s} {v:>9,}  ({v / total * 100:6.3f}%)")
    print(f"\nwrote data/cicids2017_raw.parquet ({size_mb:,.1f} MB)")
    print("wrote data/cicids2017_raw_manifest.json")
    print("\nnext: py data/build_cicids2017_flattened.py   (EDA -> flattened X/Y)")


if __name__ == "__main__":
    main()
