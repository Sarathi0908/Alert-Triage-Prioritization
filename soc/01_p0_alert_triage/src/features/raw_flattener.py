"""
raw_flattener.py  —  turn the RAW table into the flattened X/Y the model trains on.

"Flattened" here means model-ready and nothing more clever than that: one row per
flow, one column per surviving measurement, plus the two label columns. Every
transformation is a decision the raw profiler justified, and every raw column ends
up in the provenance table with the reason it was kept or dropped.

Applied, in order:
  1. drop the columns the profiler marked DROP (constant / all-zero / duplicate)
  2. repair inf -> NaN -> 0 in the columns that carry it (a zero-duration flow makes
     Flow Bytes/s divide by zero; the rate is genuinely unknown, not enormous)
  3. cast measurements to float32 (halves the table; ample precision for these)
  4. normalise the label text and derive y_is_attack (BENIGN=0, any attack=1),
     keeping the class name for per-campaign evaluation
  5. drop exact duplicate rows

Step 5 is the one that matters for honesty. CICIDS2017 contains a large number of
byte-identical flow rows; leaving them in lets the same row land in both train and
test, which inflates every metric. Dedupe before the split, not after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import raw_dataset_profiler as rp

TARGET = "y_is_attack"
ATTACK_CLASS = "attack_class"
BENIGN = "BENIGN"


def repair_inf(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """inf -> 0. A rate computed over a zero-duration flow is undefined; encoding it
    as a huge number would invent signal, and LightGBM would happily split on it."""
    if cols:
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df


def derive_labels(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(attack_class, y_is_attack) from the raw Label column."""
    cls = rp.normalise_labels(labels)
    y = (cls.str.upper() != BENIGN).astype("int8")
    return cls, y


def flatten(raw: pd.DataFrame, profile: pd.DataFrame, label_col: str,
            log=print) -> tuple[pd.DataFrame, dict]:
    """Apply the profiler's decisions. Returns (flattened, stats)."""
    keep = rp.kept_columns(profile)
    inf_cols = [c for c in rp.inf_columns(profile) if c in raw.columns]

    cls, y = derive_labels(raw[label_col])
    out = raw[keep].copy()
    out = repair_inf(out, inf_cols)

    # NaN from either the source or the inf repair becomes 0: these are counters and
    # rates where "no observation" is genuinely zero activity.
    n_nan = int(out.isna().sum().sum())
    out = out.fillna(0).astype("float32")

    out[ATTACK_CLASS] = cls.values
    out[TARGET] = y.values

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    removed = before - len(out)
    log(f"   dedupe: {before:,} -> {len(out):,} rows ({removed:,} exact duplicates removed)")

    stats = {"rows_before_dedupe": before, "rows": len(out),
             "duplicate_rows_removed": removed,
             "nan_filled": n_nan,
             "inf_repaired_columns": inf_cols,
             "n_features": len(keep), "features": keep}
    return out, stats


def provenance(profile: pd.DataFrame, kept: list[str]) -> pd.DataFrame:
    """One row per RAW column: what happened to it, and why. This is the audit that
    makes the flattened table reviewable without re-running anything."""
    keptset = set(kept)
    rows = []
    for r in profile.itertuples():
        in_model = r.raw_column in keptset
        rows.append({
            "raw_column": r.raw_column,
            "raw_position": r.position,
            "in_flattened": "YES" if in_model else "",
            "decision": r.decision,
            "reason": r.reason or ("carried through unchanged" if in_model else ""),
            "duplicate_of": r.duplicate_of,
            "null_count": r.null_count,
            "inf_count": r.pos_inf_count + r.neg_inf_count,
            "unique_finite": r.unique_finite,
            "zero_frac": r.zero_frac,
            "min": r.min, "max": r.max, "mean": r.mean,
        })
    return pd.DataFrame(rows)


def class_balance(flat: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts after flattening — dedupe hits the noisy classes hardest, so
    this is not the same distribution as the raw capture."""
    vc = flat[ATTACK_CLASS].value_counts()
    return pd.DataFrame({"attack_class": vc.index, "rows": vc.values,
                         "y": [0 if c.upper() == BENIGN else 1 for c in vc.index],
                         "share_%": (vc.values / len(flat) * 100).round(4)})
