"""
native_feature_screener.py  —  leak + NOISE screen over the native NetFlow columns
(plan §3).

Definitions, applied literally:
  NOISE  = irrelevant / redundant columns, removed by
             (a) near-constant screen (one value / almost all zero -> no information)
             (b) redundancy screen (|r| above threshold with a kept feature)
  LEAK   = a flawed column that ALONE almost perfectly separates the classes,
             removed by the single-feature ROC-AUC screen (memorisation, not behaviour)

What survives then passes a DOMINANCE screen: no feature may hold more than the
gain cap (a dominator is a leak/memorisation risk), and none may sit below the
gain floor (dead weight). The result is a small, balanced set where every
behavioural family still has a feature to detect its attack pattern by.

Thresholds live in configs/feature_config.yaml — nothing numeric is hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def family_of(col: str, families: dict[str, list[str]], default: str) -> str:
    """First family whose tokens appear in the column name (config order wins)."""
    cl = col.lower()
    for name, tokens in families.items():
        if any(str(t).lower() in cl for t in tokens):
            return name
    return default


def _ranker(params: dict, spw: float, seed: int) -> LGBMClassifier:
    """A cheap forest used only to rank/tie-break, never the final model."""
    return LGBMClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                          verbose=-1, **params)


def _gain_pct(clf: LGBMClassifier, cols: list[str]) -> pd.Series:
    g = pd.Series(clf.booster_.feature_importance("gain"), index=cols)
    total = g.sum()
    return g / total * 100 if total > 0 else g


def profile_features(sample: pd.DataFrame, y: np.ndarray, native: list[str],
                     families: dict, default_family: str) -> pd.DataFrame:
    """Per-feature statistics the screens decide on."""
    rows = []
    for c in native:
        v = sample[c].values
        a = roc_auc_score(y, v)
        rows.append({"feature": c, "family": family_of(c, families, default_family),
                     "unique": int(sample[c].nunique()),
                     "zero_frac": round(float((v == 0).mean()), 4),
                     "single_feature_auc": round(max(a, 1 - a), 4)})
    return pd.DataFrame(rows)


def screen(sample: pd.DataFrame, y: np.ndarray, native: list[str],
           feature_cfg: dict, seed: int = 42, log=print) -> tuple[list[str], pd.DataFrame, dict]:
    """Run the full screen. Returns (selected, audit_frame, payload)."""
    scr_cfg = feature_cfg["screen"]
    families, default_family = feature_cfg["families"], feature_cfg["default_family"]
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))

    scr = profile_features(sample, y, native, families, default_family)

    # ---- NOISE (a) near-constant · LEAK single-feature separation ----------------
    nc = scr_cfg["near_constant"]
    scr["near_constant"] = (scr.unique <= nc["max_unique"]) | (scr.zero_frac > nc["max_zero_frac"])
    scr["leak_suspect"] = scr.single_feature_auc >= scr_cfg["leak"]["single_feature_auc_min"]
    alive = scr[~scr.near_constant & ~scr.leak_suspect].feature.tolist()

    # ---- gain, for ranking and redundancy tie-breaks ----------------------------
    ranker_params = scr_cfg["ranker"]
    gain = _gain_pct(_ranker(ranker_params, spw, seed).fit(sample[alive], y), alive)
    scr["gain_pct"] = scr.feature.map(gain).fillna(0).round(3)

    # ---- NOISE (b) redundancy: keep the higher gain of a correlated pair --------
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

    # ---- DOMINANCE screen, then drop the near-zero tail ------------------------
    bal = scr_cfg["balance"]
    cap, floor = bal["gain_cap_pct"], bal["gain_floor_pct"]
    max_drop, min_pool, min_sel = (bal["max_dominant_drops"], bal["min_pool_size"],
                                   bal["min_selected"])
    pool, dom_dropped = list(survivors), []
    for _ in range(max_drop + 1):
        gain = _gain_pct(_ranker(ranker_params, spw, seed).fit(sample[pool], y), pool)
        if gain.max() <= cap or len(dom_dropped) >= max_drop or len(pool) <= min_pool:
            break
        dom = gain.idxmax()
        dom_dropped.append(dom)
        pool.remove(dom)
        log(f"   dominance screen: dropped {dom} ({gain.max():.0f}% gain), refit")

    selected = [c for c in pool if gain.get(c, 0) >= floor]
    if len(selected) < min_sel:      # keep the strongest few even if they dip below
        selected = sorted(pool, key=lambda c: -gain.get(c, 0))[:min(min_sel, len(pool))]
    low_tail = [c for c in pool if c not in selected]
    selected = sorted(selected, key=lambda c: -gain.get(c, 0))

    # ---- audit: every dropped column carries a reason --------------------------
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
    auc_by_feature = scr.set_index("feature").single_feature_auc
    payload = {"features": selected, "n": len(selected),
               "families": {c: family_of(c, families, default_family) for c in selected},
               "gain_pct": {c: round(float(gain.get(c, 0)), 3) for c in selected},
               "single_feature_auc": {c: float(auc_by_feature[c]) for c in selected},
               "removed": removed, "categorical": []}
    return selected, scr.sort_values("gain_pct", ascending=False), payload


def log_summary(payload: dict, feature_cfg: dict, log=print) -> None:
    bal = feature_cfg["screen"]["balance"]
    sel, gains = payload["features"], payload["gain_pct"]
    log(f"\nSELECTED {len(sel)} native features:")
    for c in sel:
        log(f"   {gains[c]:5.1f}%  auc {payload['single_feature_auc'][c]:.2f}  "
            f"[{payload['families'][c]:13s}] {c}")
    g = sorted(gains.values())
    log(f"\nBALANCE: max gain {max(g):.1f}%  min gain {min(g):.1f}%  "
        f"(target [{bal['gain_floor_pct']}%, {bal['gain_cap_pct']}%])")
    for reason, cols in payload["removed"].items():
        log(f"removed as {reason}: {len(cols)}")
