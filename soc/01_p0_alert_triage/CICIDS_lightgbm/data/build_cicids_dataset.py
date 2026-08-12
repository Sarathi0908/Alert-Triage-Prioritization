"""
build_cicids_dataset.py — STEP 1: merge CICIDS2017 (MachineLearningCVE, 8 days) into ONE clean table.

Source: Downloads/MachineLearningCSV.zip  (the official CIC MachineLearningCVE set: 78 CICFlowMeter
flow features + Label, no Source-IP identity columns).

Cleaning: strip the leading spaces CIC leaves in column names, replace +/-inf with NaN, drop NaN rows
and exact duplicates. Label: BENIGN -> y_is_attack 0, any attack -> 1; the original class name is kept
as attack_class for the per-class EDA graphs.

Outputs:
  data/cicids2017_merged.parquet          (X + y + attack_class + _day)
  data/cicids2017_feature_manifest.json   (native feature list + label map)
  Downloads/CICIDS2017_merged_raw_sample.csv   (raw sheet, 2000 rows, small)
"""
import os, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
# Where the downloaded benchmark archive lives. Override for a different machine:
#   set BHAIRAVA_DOWNLOADS=D:\datasets
DL = Path(os.getenv("BHAIRAVA_DOWNLOADS", "~/Downloads")).expanduser()
ZIP = DL / "MachineLearningCSV.zip"

print(f"reading {ZIP.name} ...", flush=True)
frames = []
with zipfile.ZipFile(ZIP) as z:
    for n in sorted(x for x in z.namelist() if x.lower().endswith(".csv")):
        with z.open(n) as f:
            d = pd.read_csv(f, low_memory=False)
        d.columns = d.columns.str.strip()
        d["_day"] = Path(n).name.split(".pcap")[0]
        frames.append(d)
        print(f"  {Path(n).name}: {len(d):,} rows", flush=True)

df = pd.concat(frames, ignore_index=True)
df.columns = df.columns.str.strip()
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates().reset_index(drop=True)
print(f"\nmerged {before:,} -> {len(df):,} rows after inf->NaN, dropna, dedup", flush=True)

# ---- label ----
df["Label"] = df["Label"].astype(str).str.strip()
df["attack_class"] = df["Label"]
df["y_is_attack"] = (df["Label"].str.upper() != "BENIGN").astype(int)

NATIVE = [c for c in df.columns if c not in ("Label", "attack_class", "y_is_attack", "_day")]
# force numeric on features
for c in NATIVE:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=NATIVE).reset_index(drop=True)

print(f"\nATTACK RATE: {df.y_is_attack.mean()*100:.3f}%  ({int(df.y_is_attack.sum()):,} attacks)")
print("\nATTACK CLASSES (all patterns present across the 8 days):")
print(df.attack_class.value_counts().to_string())
print(f"\nnative features: {len(NATIVE)}")

df.to_parquet(HERE / "cicids2017_merged.parquet", index=False)
json.dump({"rows": int(len(df)), "native_features": NATIVE, "n_native": len(NATIVE),
           "target": "y_is_attack", "attack_class_col": "attack_class",
           "label_map": "BENIGN=0, any attack=1",
           "attack_rate_%": round(float(df.y_is_attack.mean() * 100), 4),
           "attack_classes": df.attack_class.value_counts().to_dict()},
          open(HERE / "cicids2017_feature_manifest.json", "w"), indent=2)
df.head(2000).to_csv(DL / "CICIDS2017_merged_raw_sample.csv", index=False)
print(f"\nwrote data/cicids2017_merged.parquet ({len(df):,} x {len(NATIVE)} native)")
print(f"wrote Downloads/CICIDS2017_merged_raw_sample.csv (raw sheet, 2000 rows)")
