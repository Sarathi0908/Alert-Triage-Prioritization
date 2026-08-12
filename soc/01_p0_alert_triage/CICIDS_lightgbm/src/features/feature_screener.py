"""
feature_screener.py  —  leak + NOISE screen over the native NetFlow features.

Definitions, applied literally:
  NOISE  = irrelevant / redundant columns -> removed by
             (a) near-constant screen (one value / almost all zero -> no information)
             (b) redundancy screen (|r| > threshold with a kept feature -> duplicate signal)
  LEAK   = a flawed column that ALONE almost perfectly separates the classes ->
             removed by the single-feature ROC-AUC screen (AUC alone >= threshold
             is memorisation, not behaviour)

After removing noise + leaks, keep the features that survive a DOMINANCE screen
(no single feature may hold more than the gain cap; none below the gain floor) so
importance stays balanced and every behavioural family keeps coverage.

Thresholds live in configs/feature_config.yaml — nothing numeric is hardcoded here.

Writes: data/native_selected.json  +  data/feature_screen.csv (the full audit).
Run:  py src/features/feature_screener.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import load_config  # noqa: E402
from src import data_source as ds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def family_of(col: str, families: dict[str, list[str]], default: str) -> str:
    """First family whose tokens appear in the column name (config order wins)."""
    cl = col.lower()
    for name, tokens in families.items():
        if any(str(t).lower() in cl for t in tokens):
            return name
    return default


def _ranker(params: dict, spw: float, seed: int) -> LGBMClassifier:
    return LGBMClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1, **params)


def _gain_pct(clf: LGBMClassifier, cols: list[str]) -> pd.Series:
    g = pd.Series(clf.booster_.feature_importance("gain"), index=cols)
    total = g.sum()
    return g / total * 100 if total > 0 else g


def main() -> None:
    mcfg, fcfg = load_config("model"), load_config("feature")
    seed = mcfg["seed"]
    scr_cfg = fcfg["screen"]
    target = mcfg["target"]

    exclude = set(fcfg.get("exclude") or [])
    native = [c for c in ds.native_features(mcfg) if c not in exclude]

    frame = ds.load_frame(mcfg, columns=native, with_class=False)
    sample = ds.subsample(frame, scr_cfg["sample_rows"], seed)
    y = sample[target].astype(int).values
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"screening {len(native)} native features on {len(sample):,} rows "
          f"(attack {y.mean() * 100:.2f}%)", flush=True)

    # ---- per-feature statistics -------------------------------------------------
    families, default_family = fcfg["families"], fcfg["default_family"]
    rows = []
    for c in native:
        v = sample[c].values
        a = roc_auc_score(y, v)
        rows.append({"feature": c, "family": family_of(c, families, default_family),
                     "unique": int(sample[c].nunique()),
                     "zero_frac": round(float((v == 0).mean()), 4),
                     "single_feature_auc": round(max(a, 1 - a), 4)})
    scr = pd.DataFrame(rows)

    # ---- NOISE (a) near-constant · LEAK single-feature separation ----------------
    nc = scr_cfg["near_constant"]
    scr["near_constant"] = (scr.unique <= nc["max_unique"]) | (scr.zero_frac > nc["max_zero_frac"])
    scr["leak_suspect"] = scr.single_feature_auc >= scr_cfg["leak"]["single_feature_auc_min"]
    alive = scr[~scr.near_constant & ~scr.leak_suspect].feature.tolist()

    # ---- gain, for ranking and redundancy tie-breaks ----------------------------
    ranker_params = scr_cfg["ranker"]
    gain = _gain_pct(_ranker(ranker_params, spw, seed).fit(sample[alive], y), alive)
    scr["gain_pct"] = scr.feature.map(gain).fillna(0).round(3)

    # ---- NOISE (b) redundancy: |r| above threshold, keep the higher gain --------
    corr = sample[alive].corr().abs()
    r_max = scr_cfg["redundancy"]["abs_corr_max"]
    dropped: set[str] = set()
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a_, b_ = alive[i], alive[j]
            if a_ in dropped or b_ in dropped:
                continue
            if corr.iloc[i, j] > r_max:
                dropped.add(a_ if gain.get(a_, 0) < gain.get(b_, 0) else b_)
    survivors = [c for c in alive if c not in dropped]

    # ---- DOMINANCE screen: drop extreme dominators, then the near-zero tail -----
    bal = scr_cfg["balance"]
    cap, floor = bal["gain_cap_pct"], bal["gain_floor_pct"]
    max_drop, min_pool, min_sel = bal["max_dominant_drops"], bal["min_pool_size"], bal["min_selected"]
    pool, dom_dropped = list(survivors), []
    for _ in range(max_drop + 1):
        gain = _gain_pct(_ranker(ranker_params, spw, seed).fit(sample[pool], y), pool)
        if gain.max() <= cap or len(dom_dropped) >= max_drop or len(pool) <= min_pool:
            break
        dom = gain.idxmax()
        dom_dropped.append(dom)
        pool.remove(dom)
        print(f"   dominance screen: dropped {dom} ({gain.max():.0f}% gain), refit", flush=True)

    selected = [c for c in pool if gain.get(c, 0) >= floor]
    if len(selected) < min_sel:   # keep the strongest few even if they dip below the floor
        selected = sorted(pool, key=lambda c: -gain.get(c, 0))[:min(min_sel, len(pool))]
    low_tail = [c for c in pool if c not in selected]
    selected = sorted(selected, key=lambda c: -gain.get(c, 0))

    # ---- audit ------------------------------------------------------------------
    leak_key = f"leak_suspect (auc>={scr_cfg['leak']['single_feature_auc_min']})"
    redundant_key = f"redundant_noise (|r|>{r_max})"
    removed = {"near_constant_noise": scr[scr.near_constant].feature.tolist(),
               leak_key: scr[scr.leak_suspect].feature.tolist(),
               redundant_key: sorted(dropped),
               "dominant_removed_for_balance": dom_dropped,
               "low_gain_tail": sorted(low_tail)}

    def role_of(f: str) -> str:
        if f in selected:
            return "SELECTED"
        if f in removed["near_constant_noise"]:
            return "near-constant NOISE"
        if f in removed[leak_key]:
            return "LEAK"
        if f in removed[redundant_key]:
            return "redundant NOISE"
        if f in removed["dominant_removed_for_balance"]:
            return "DOMINANT (removed for balance)"
        if f in removed["low_gain_tail"]:
            return "low-gain tail"
        return "not-selected"

    scr["role"] = scr.feature.apply(role_of)
    audit = ds.data_dir(mcfg) / mcfg["data"]["screen_audit"]
    scr.sort_values("gain_pct", ascending=False).to_csv(audit, index=False)

    auc_by_feature = scr.set_index("feature").single_feature_auc
    out = ds.save_selected(mcfg, {
        "features": selected, "n": len(selected),
        "families": {c: family_of(c, families, default_family) for c in selected},
        "gain_pct": {c: round(float(gain.get(c, 0)), 3) for c in selected},
        "single_feature_auc": {c: float(auc_by_feature[c]) for c in selected},
        "removed": removed,
        "categorical": [],
    })

    print(f"\nSELECTED {len(selected)} native features:")
    for c in selected:
        print(f"   {gain.get(c, 0):5.1f}%  auc {auc_by_feature[c]:.2f}  "
              f"[{family_of(c, families, default_family):13s}] {c}")
    gsel = sorted(gain.get(c, 0) for c in selected)
    print(f"\nBALANCE: max gain {max(gsel):.1f}%  min gain {min(gsel):.1f}%  "
          f"(target [{floor}%, {cap}%])")
    print(f"removed as DOMINANT (for balance): {removed['dominant_removed_for_balance']}")
    print(f"removed as low-gain tail: {len(removed['low_gain_tail'])}")
    print(f"removed as NOISE (near-constant): {len(removed['near_constant_noise'])}")
    print(f"removed as NOISE (redundant |r|>{r_max}): {len(removed[redundant_key])}")
    print(f"removed as LEAK: {removed[leak_key]}")
    print(f"wrote {out.relative_to(_SLOT_ROOT)} + {audit.relative_to(_SLOT_ROOT)}")


if __name__ == "__main__":
    main()
