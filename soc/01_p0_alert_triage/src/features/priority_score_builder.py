"""
priority_score_builder.py  —  the graded 0-100 analyst priority (plan §2/§6).

A binary ALERT/NORMAL verdict cliffs the analyst queue: everything above the
threshold looks equally urgent. Priority restores an ordering by blending model
confidence with traffic volume, each mapped to its PERCENTILE so the two
incomparable scales combine meaningfully.

Two rules the rest of the slot depends on:
  * priority is computed over PREDICTED ALERTS ONLY — normal traffic is never
    ranked, so a benign flow has no priority and no decile;
  * the percentile grids are captured at fit time and travel in the model bundle,
    so serving reproduces training's ranking instead of re-deriving it per request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_QUANTILES = 101          # percentile grid resolution stored in the bundle


class PriorityScoreBuilder:
    def __init__(self, weights: dict, volume_feature: str, n_deciles: int = 10,
                 log1p_volume: bool = True):
        self.w_prob = float(weights["probability"])
        self.w_vol = float(weights["volume"])
        self.volume_feature = volume_feature
        self.n_deciles = int(n_deciles)
        self.log1p_volume = bool(log1p_volume)
        self.p_grid: list[float] = []
        self.vol_grid: list[float] = []
        self.rank_q: list[float] = []
        self.decile_edges: list[float] | None = None

    # ---- fit ----------------------------------------------------------------
    def volume_of(self, frame: pd.DataFrame) -> np.ndarray:
        raw = pd.to_numeric(frame[self.volume_feature], errors="coerce").fillna(0).values.astype(float)
        return np.log1p(raw) if self.log1p_volume else raw

    def fit(self, alert_probs: np.ndarray, alert_volume: np.ndarray) -> "PriorityScoreBuilder":
        """Capture the training-time percentile grids from the ALERTS only."""
        q = np.linspace(0, 1, _QUANTILES)
        self.rank_q = [float(x) for x in q]
        self.p_grid = [float(x) for x in np.quantile(alert_probs, q)]
        self.vol_grid = [float(x) for x in np.quantile(alert_volume, q)]
        pri = self.transform_ranked(alert_probs, alert_volume)
        self.decile_edges = ([float(x) for x in
                              np.quantile(pri, np.linspace(0, 1, self.n_deciles + 1))]
                             if len(pri) else None)
        return self

    # ---- transform ----------------------------------------------------------
    def transform_ranked(self, probs: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """In-sample priority using rank percentiles (used at fit time, where the
        whole alert population is in hand)."""
        return 100.0 * (self.w_prob * pd.Series(probs).rank(pct=True).values
                        + self.w_vol * pd.Series(volume).rank(pct=True).values)

    def _pct(self, values, grid) -> np.ndarray:
        if not grid:
            return np.zeros_like(np.asarray(values, dtype=float))
        return np.interp(np.asarray(values, dtype=float), np.asarray(grid),
                         np.asarray(self.rank_q))

    def transform(self, probs, volume) -> np.ndarray:
        """Serving-time priority: interpolate against the stored grids, so a single
        request gets the same percentile it would have had in training."""
        return 100.0 * (self.w_prob * self._pct(probs, self.p_grid)
                        + self.w_vol * self._pct(volume, self.vol_grid))

    def decile(self, priority) -> np.ndarray:
        """1..n_deciles, highest = act first."""
        priority = np.asarray(priority, dtype=float)
        if self.decile_edges:
            inner = np.array(self.decile_edges[1:-1])
            return np.clip(np.searchsorted(inner, priority, side="right") + 1, 1, self.n_deciles)
        return np.clip((priority / (100 / self.n_deciles)).astype(int) + 1, 1, self.n_deciles)

    # ---- bundle round-trip ---------------------------------------------------
    def to_bundle(self) -> dict:
        return {"vol_feature": self.volume_feature,
                "priority_weights": {"probability": self.w_prob, "volume": self.w_vol},
                "p_grid": self.p_grid, "vol_grid": self.vol_grid,
                "rank_q": self.rank_q, "decile_edges": self.decile_edges,
                "log1p_volume": self.log1p_volume, "n_deciles": self.n_deciles}

    @classmethod
    def from_bundle(cls, blob: dict) -> "PriorityScoreBuilder":
        w = blob.get("priority_weights", {"probability": 0.70, "volume": 0.30})
        obj = cls(w, blob.get("vol_feature", ""), blob.get("n_deciles", 10),
                  blob.get("log1p_volume", True))
        obj.p_grid = blob.get("p_grid") or []
        obj.vol_grid = blob.get("vol_grid") or []
        obj.rank_q = blob.get("rank_q") or []
        obj.decile_edges = blob.get("decile_edges")
        return obj

    def formula(self) -> str:
        vol = (f"log1p({self.volume_feature})" if self.log1p_volume else self.volume_feature)
        return f"priority = 100*({self.w_prob}*pct(prob) + {self.w_vol}*pct({vol}))"
