"""
raw_dataset_profiler.py  —  EDA over the RAW table, and the flattening decisions
it produces (plan §3).

This is the stage between raw and model-ready. It profiles every raw column and
turns what it finds into an explicit decision per column, so the flattened table is
a consequence of evidence rather than of habit:

  DROP  constant           one distinct finite value -> no information to learn from
  DROP  all-zero           every value 0 -> same thing, stated separately because it
                           is the common CICFlowMeter failure mode (bulk-rate fields)
  DROP  duplicate          byte-identical to an earlier column (CICIDS2017 ships
                           `Fwd Header Length.1`, an exact copy)
  KEEP  repair-inf         real signal, but carries inf/-inf (Flow Bytes/s and
                           Flow Packets/s divide by a zero duration) -> repaired
  KEEP                     everything else

Memory: one column at a time via pyarrow, so a 2.8M-row table profiles in ~25 MB
rather than 1.8 GB.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation import plots as pl

# Decision codes, in the order they are applied.
DROP_CONSTANT = "DROP constant"
DROP_ALL_ZERO = "DROP all-zero"
DROP_DUPLICATE = "DROP duplicate column"
KEEP_REPAIR_INF = "KEEP (repair inf)"
KEEP = "KEEP"


def _column(path, name: str) -> np.ndarray:
    """Read a single column as float64 without materialising the table."""
    return pq.read_table(path, columns=[name])[name].to_numpy(zero_copy_only=False)


def _fingerprint(v: np.ndarray) -> str:
    """Stable hash of a column's values, NaN-normalised so two all-NaN columns match."""
    b = np.nan_to_num(v, nan=np.nan, posinf=np.inf, neginf=-np.inf)
    return hashlib.sha1(np.ascontiguousarray(b, dtype="float64").tobytes()).hexdigest()


def profile_columns(raw_path, raw_columns: list[str], log=print) -> pd.DataFrame:
    """One row per raw column: counts, range, and the reason it will be kept/dropped."""
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for i, name in enumerate(raw_columns, 1):
        v = _column(raw_path, name)
        n = len(v)
        nan = int(np.isnan(v).sum())
        pos_inf = int(np.isposinf(v).sum())
        neg_inf = int(np.isneginf(v).sum())
        finite = v[np.isfinite(v)]
        n_unique_finite = int(pd.unique(finite).size) if finite.size else 0
        zero_frac = float((finite == 0).mean()) if finite.size else 1.0

        fp = _fingerprint(v)
        duplicate_of = seen.get(fp)
        if duplicate_of is None:
            seen[fp] = name

        if duplicate_of is not None:
            decision, why = DROP_DUPLICATE, f"byte-identical to `{duplicate_of}`"
        elif n_unique_finite <= 1 and zero_frac == 1.0:
            decision, why = DROP_ALL_ZERO, "every value is 0"
        elif n_unique_finite <= 1:
            decision, why = DROP_CONSTANT, "one distinct finite value"
        elif pos_inf or neg_inf:
            decision, why = KEEP_REPAIR_INF, f"{pos_inf + neg_inf:,} inf values to repair"
        else:
            decision, why = KEEP, ""

        rows.append({
            "raw_column": name, "position": i - 1,
            "null_count": nan, "pos_inf_count": pos_inf, "neg_inf_count": neg_inf,
            "unique_finite": n_unique_finite,
            "zero_frac": round(zero_frac, 6),
            "min": round(float(finite.min()), 4) if finite.size else "",
            "max": round(float(finite.max()), 4) if finite.size else "",
            "mean": round(float(finite.mean()), 4) if finite.size else "",
            "decision": decision, "reason": why, "duplicate_of": duplicate_of or "",
        })
        if i % 20 == 0:
            log(f"   profiled {i}/{len(raw_columns)} columns")
        del v, finite
    return pd.DataFrame(rows)


def normalise_labels(s: pd.Series) -> pd.Series:
    """CIC ships the Web Attack labels with a mojibake byte where an en-dash belongs
    ('Web Attack � Brute Force'). Normalise it so the class name is addressable."""
    return (s.astype(str)
            .str.replace("�", "-", regex=False)
            .str.replace("�", "-", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def kept_columns(profile: pd.DataFrame) -> list[str]:
    return profile.loc[profile.decision.str.startswith("KEEP"), "raw_column"].tolist()


def inf_columns(profile: pd.DataFrame) -> list[str]:
    return profile.loc[profile.decision == KEEP_REPAIR_INF, "raw_column"].tolist()


def summarise(profile: pd.DataFrame) -> dict:
    counts = profile.decision.value_counts().to_dict()
    return {"n_raw_columns": int(len(profile)),
            "n_kept": int(len(kept_columns(profile))),
            "decisions": counts,
            "dropped": {
                row.raw_column: row.reason
                for row in profile[profile.decision.str.startswith("DROP")].itertuples()},
            "inf_repaired": inf_columns(profile)}


# ---- EDA charts over the raw table -------------------------------------------

def chart_class_counts(label_counts: dict, out, log=print):
    """Every campaign in the capture, log scale — the imbalance is the headline."""
    try:
        plt = pl._mpl()
        s = pd.Series({k: v for k, v in label_counts.items() if k.upper() != "BENIGN"})
        s = s.sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(s.index, s.values, color=pl.RED, height=.72)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=pl.INK2)
        ax.set_xscale("log")
        pl.style(ax, "CICIDS2017 attack classes in the RAW capture (log scale)", "flows")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot raw class_counts failed: {exc}")
        return None


def chart_data_quality(profile: pd.DataFrame, out, log=print):
    """Why the flattening drops what it drops: nulls, infs and dead columns."""
    try:
        plt = pl._mpl()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # Boolean masks, not .query(): in pandas' expression parser `inf` is the
        # float-infinity literal, so query("inf > 0") collapses to a scalar True.
        total_inf = profile.pos_inf_count + profile.neg_inf_count
        infs = (profile.assign(inf_total=total_inf)
                .loc[total_inf > 0].sort_values("inf_total"))
        if len(infs):
            axes[0].barh(infs.raw_column, infs.inf_total, color=pl.RED, height=.6)
            for i, v in enumerate(infs.inf_total):
                axes[0].text(v, i, f" {int(v):,}", va="center", fontsize=8, color=pl.INK2)
        else:
            axes[0].text(.5, .5, "no inf values", ha="center", va="center",
                         transform=axes[0].transAxes, color=pl.INK2)
        pl.style(axes[0], f"Columns carrying inf ({len(infs)})", "inf values")

        nulls = profile.loc[profile.null_count > 0].sort_values("null_count")
        if len(nulls):
            axes[1].barh(nulls.raw_column, nulls.null_count, color=pl.RED, height=.6)
        else:
            axes[1].text(.5, .5, "no nulls", ha="center", va="center",
                         transform=axes[1].transAxes, color=pl.INK2)
        pl.style(axes[1], f"Columns carrying nulls ({len(nulls)})", "null values")

        dec = profile.decision.value_counts().sort_values()
        axes[2].barh(dec.index, dec.values,
                     color=[pl.RED if k.startswith("DROP") else pl.BLUE for k in dec.index],
                     height=.6)
        for i, v in enumerate(dec.values):
            axes[2].text(v, i, f" {v}", va="center", fontsize=9, color=pl.INK2)
        pl.style(axes[2], "Flattening decision per raw column", "columns")

        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot raw data_quality failed: {exc}")
        return None


def chart_attack_rate_by_day(day_attack: pd.DataFrame, out, log=print):
    """Attack rate per capture day — shows why a temporal split would be wrong here:
    whole campaigns live on single days."""
    try:
        plt = pl._mpl()
        d = day_attack.sort_values("attack_rate")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.barh(d._day, d.attack_rate * 100, color=pl.AQUA, height=.6)
        for i, r in enumerate(d.itertuples()):
            ax.text(r.attack_rate * 100, i, f"  {r.attack_rate * 100:.1f}%  (n={r.rows:,})",
                    va="center", fontsize=8, color=pl.INK2)
        ax.set_xlim(0, max(60, d.attack_rate.max() * 130))
        pl.style(ax, "Attack rate by capture day", "% of flows that are attacks")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
    except Exception as exc:
        log(f"plot raw attack_rate_by_day failed: {exc}")
        return None
