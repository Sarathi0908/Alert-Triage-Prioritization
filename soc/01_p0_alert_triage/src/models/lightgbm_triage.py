"""
lightgbm_triage.py  —  LightGBM triage classifier for SOC alert triage (plan §4).

Learns P(attack | native NetFlow measurements) on the leak/noise-screened feature
set. That probability IS the triage score: the BACKEND routes NORMAL to the Data
Lake and ALERT to the analyst queue.

Two things travel with the fitted model so the served operating point can never
drift from the evaluated one:
  * `threshold` — fitted, not fixed: the highest-recall point that still holds the
    configured precision floor (`fit_threshold`).
  * the percentile grids the priority blend interpolates against
    (src/features/priority_score_builder.py owns that arithmetic).

`attack_probability` returns higher = more likely an attack, so it feeds straight
into the priority score and the alert queue.
"""
from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


class LightGBMTriage:
    def __init__(self, cfg: dict, seed: int = 42):
        self.params = dict(cfg)
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.5
        self.scale_pos_weight: float = 1.0

    # ---- fit ----------------------------------------------------------------
    def fit(self, X, feature_names: list[str], y) -> "LightGBMTriage":
        """Fit on the training split. `scale_pos_weight` is computed here rather
        than configured, because it is a property of the split, not a choice."""
        self.feature_names = list(feature_names)
        y = np.asarray(y).astype(int)
        self.scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        self.model = lgb.LGBMClassifier(scale_pos_weight=self.scale_pos_weight,
                                        random_state=self.seed, **self.params)
        self.model.fit(self._frame(X), y)
        return self

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        return pd.DataFrame(np.asarray(X), columns=self.feature_names)

    # ---- score --------------------------------------------------------------
    def attack_probability(self, X) -> np.ndarray:
        """Higher = more likely an attack. Calibrated probability, directly rankable."""
        return self.model.predict_proba(self._frame(X))[:, 1].astype("float64")

    def fit_threshold(self, y_true, scores, min_precision: float,
                      fallback: float = 0.5) -> float:
        """Highest-recall threshold that still holds precision >= the floor.

        A fixed 0.5 would leave the analyst queue's precision to chance; the queue
        is precision-constrained, so the floor is what we hold and recall is what
        we maximise under it.
        """
        prc, rec, thr = precision_recall_curve(y_true, scores)
        ok = np.where(prc[:-1] >= min_precision)[0]
        self.threshold = float(thr[ok[np.argmax(rec[:-1][ok])]]) if len(ok) else float(fallback)
        return self.threshold

    def verdict(self, scores) -> np.ndarray:
        """1 = ALERT (analyst queue), 0 = NORMAL (Data Lake)."""
        return (np.asarray(scores) >= self.threshold).astype(int)

    # ---- explain ------------------------------------------------------------
    def shap_contributions(self, X) -> np.ndarray:
        """Exact tree SHAP from LightGBM's own `pred_contrib` — no `shap` package.
        Returns one row per sample, one column per feature (bias term dropped)."""
        return self.model.booster_.predict(self._frame(X), pred_contrib=True)[:, :-1]

    def gain_importance(self) -> pd.Series:
        """Split-gain importance as a percentage of total gain."""
        g = pd.Series(self.model.booster_.feature_importance("gain"),
                      index=self.feature_names)
        total = g.sum()
        return (g / total * 100) if total > 0 else g

    # ---- persist ------------------------------------------------------------
    def save(self, path: str, extra: dict | None = None) -> None:
        blob = {"model": self.model, "features": self.feature_names,
                "native": self.feature_names, "cats": [], "cat_levels": {},
                "threshold": self.threshold,
                "scale_pos_weight": self.scale_pos_weight,
                "params": self.params, "seed": self.seed}
        blob.update(extra or {})
        joblib.dump(blob, path)

    @classmethod
    def load(cls, path: str) -> tuple["LightGBMTriage", dict]:
        """Return (model, blob). The blob carries the serving extras the API needs
        (priority grids, volume feature, decile edges)."""
        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = blob["model"]
        obj.feature_names = blob["features"]
        obj.threshold = blob["threshold"]
        obj.scale_pos_weight = blob.get("scale_pos_weight", 1.0)
        obj.params = blob.get("params", {})
        obj.seed = blob.get("seed", 42)
        return obj, blob
