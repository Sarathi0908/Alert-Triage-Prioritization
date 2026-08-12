"""
build_ids2018_dataset.py — STEP 1: load NF-CSE-CIC-IDS2018-v2 (NetFlow) into one clean table.

Source: Downloads/b3427ed8ad063a09_MOHANAD_A4706.zip -> NF-CSE-CIC-IDS2018-v2.csv (18.9M flows).
Feature engineering at load time:
  * DROP identity / leak columns: IPV4_SRC_ADDR, IPV4_DST_ADDR (IP identity),
    L4_SRC_PORT (ephemeral source port = noise), DNS_QUERY_ID (random transaction id).
  * Keep the remaining NetFlow measurements as NATIVE features.
Label: `Label` is already 0/1 (attack/benign); the class name is kept as attack_class (`Attack`).
Chunked read + ~27% stratified-random subsample -> ~5M rows (tractable, natural distribution).

Outputs:
  data/ids2018_merged.parquet
  data/ids2018_feature_manifest.json
  Downloads/CSE_CIC_IDS2018_merged_raw_sample.csv (raw sheet, 2000 rows)
"""
import os, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parent
# Where the downloaded benchmark archive lives. Override for a different machine:
#   set BHAIRAVA_DOWNLOADS=D:\datasets
DL = Path(os.getenv("BHAIRAVA_DOWNLOADS", "~/Downloads")).expanduser()
ZIP = DL / "b3427ed8ad063a09_MOHANAD_A4706.zip"
CSV = "b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv"
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "DNS_QUERY_ID"]  # identity / ephemeral / random
KEEP = 0.27
rng = np.random.RandomState(42)

print("reading NF-CSE-CIC-IDS2018-v2 in chunks ...", flush=True)
parts, seen = [], 0
with zipfile.ZipFile(ZIP) as z, z.open(CSV) as f:
    for ch in pd.read_csv(f, chunksize=2_000_000, low_memory=False):
        seen += len(ch)
        ch = ch.drop(columns=[c for c in DROP if c in ch.columns])
        parts.append(ch[rng.random(len(ch)) < KEEP])
        print(f"  read {seen:,} ...", flush=True)
df = pd.concat(parts, ignore_index=True)
print(f"subsampled to {len(df):,} rows", flush=True)

NATIVE = [c for c in df.columns if c not in ("Label", "Attack")]
for c in NATIVE:
    df[c] = pd.to_numeric(df[c], errors="coerce")
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=NATIVE).drop_duplicates().reset_index(drop=True)
df["y_is_attack"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
df["attack_class"] = df["Attack"].astype(str).str.strip()
print(f"clean {before:,} -> {len(df):,}")
print(f"\nATTACK RATE: {df.y_is_attack.mean()*100:.3f}%")
print("\nATTACK CLASSES:")
print(df.attack_class.value_counts().to_string())

df.to_parquet(HERE / "ids2018_merged.parquet", index=False)
json.dump({"dataset": "CSE-CIC-IDS2018 (NF-v2, NetFlow)", "rows": int(len(df)),
           "native_features": NATIVE, "n_native": len(NATIVE), "target": "y_is_attack",
           "attack_class_col": "attack_class", "label_map": "Label 0/1 (benign/attack)",
           "dropped_identity": DROP, "attack_rate_%": round(float(df.y_is_attack.mean() * 100), 4),
           "attack_classes": df.attack_class.value_counts().to_dict()},
          open(HERE / "ids2018_feature_manifest.json", "w"), indent=2)
df.head(2000).to_csv(DL / "CSE_CIC_IDS2018_merged_raw_sample.csv", index=False)
print(f"\nwrote data/ids2018_merged.parquet ({len(df):,} x {len(NATIVE)} native)")
print("wrote data/ids2018_feature_manifest.json + Downloads/CSE_CIC_IDS2018_merged_raw_sample.csv")
