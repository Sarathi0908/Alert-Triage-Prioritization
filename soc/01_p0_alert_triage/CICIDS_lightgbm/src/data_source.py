"""
data_source.py  —  load the merged NetFlow table for Alert Triage (#01,
                   CSE-CIC-IDS2018 variant).

One row = one NetFlow record. The table is produced by
data/build_ids2018_dataset.py (identity/ephemeral/random columns already dropped
at build time) and is gitignored — regenerate it, never commit it.

Every path comes from configs/model_config.yaml `data.*`, resolved slot-relative,
so nothing here hardcodes a machine path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path

# Values outside float32 range appear in a few NetFlow rate columns; clip rather
# than overflow to inf when casting.
_F32_CLIP = 3e38


def data_dir(model_cfg: dict) -> Path:
    return Path(resolve_path(model_cfg["data"]["dir"]))


def dataset_path(model_cfg: dict) -> Path:
    """Absolute path to the merged parquet the model trains on."""
    return data_dir(model_cfg) / model_cfg["data"]["dataset"]


def load_manifest(model_cfg: dict) -> dict:
    """The dataset build manifest: native feature list, label map, class counts."""
    with open(data_dir(model_cfg) / model_cfg["data"]["manifest"], encoding="utf-8") as fh:
        return json.load(fh)


def native_features(model_cfg: dict) -> list[str]:
    """Every native measurement column present in the built table."""
    return list(load_manifest(model_cfg)["native_features"])


def load_selected(model_cfg: dict) -> dict:
    """The screened model feature set written by src/features/feature_screener.py."""
    with open(data_dir(model_cfg) / model_cfg["data"]["selected"], encoding="utf-8") as fh:
        return json.load(fh)


def save_selected(model_cfg: dict, payload: dict) -> Path:
    path = data_dir(model_cfg) / model_cfg["data"]["selected"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Force the feature columns to clean float32: numeric, clipped, no inf/NaN.

    LightGBM tolerates NaN but the correlation/AUC screens and the SHAP charts do
    not, so every consumer gets the same well-defined matrix.
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(-_F32_CLIP, _F32_CLIP).astype(np.float32)
    return df.replace([np.inf, -np.inf], 0).fillna(0)


def clean_attack_class(s: pd.Series) -> pd.Series:
    """Normalise the class name — the published CSV carries a mojibake dash."""
    return s.astype(str).str.replace("�", "-", regex=False).str.strip()


def load_frame(model_cfg: dict, columns: list[str] | None = None,
               with_class: bool = True) -> pd.DataFrame:
    """Load the merged table (optionally a column subset) with features cleaned.

    `columns` are feature columns; the target and attack_class are appended
    automatically so callers cannot accidentally drop the label.
    """
    target = model_cfg["target"]
    cls = model_cfg["attack_class_column"]
    if columns is None:
        df = pd.read_parquet(dataset_path(model_cfg))
        feats = [c for c in df.columns if c not in (target, cls, "Label", "Attack", "_day")]
    else:
        feats = list(columns)
        want = feats + [target] + ([cls] if with_class else [])
        df = pd.read_parquet(dataset_path(model_cfg), columns=want)
    df = coerce_numeric(df, feats)
    if cls in df.columns:
        df[cls] = clean_attack_class(df[cls])
    return df


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random subsample used by the screens (the full table exceeds what they need).

    Draws without replacement even when `n` covers the whole table, which yields a
    permutation rather than the original row order. That is deliberate, not a
    quirk: the screening ranker runs without LightGBM's `deterministic` /
    `force_row_wise` flags, so its threaded histogram sums are mildly
    order-sensitive and short-circuiting here would shift the reported gain
    percentages by ~0.1pp against the reference run.
    """
    idx = np.random.RandomState(seed).choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def pick_volume_feature(features: list[str], preference: list[str]) -> str:
    """The traffic-volume term in the priority blend: first preference that matches
    a feature name (case-insensitive substring), else the first feature."""
    for token in preference:
        for f in features:
            if token.upper() in f.upper():
                return f
    return features[0]
