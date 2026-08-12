"""
api.py — serving layer for the Alert Triage LightGBM model (NATIVE-only).

REQUEST BODY = a JSON object of the model's NATIVE NetFlow features, by their real
names (e.g. "L4_DST_PORT", "SRC_TO_DST_AVG_THROUGHPUT"). No derived features — the
body IS the feature vector. Missing features default to 0. The feature list, the
alert threshold and the priority grids all come from the model bundle, so the
served contract and operating point can never drift from what was evaluated.

Run:  py src/serving/api.py          (or: py -m src.serving.api from the slot root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import OUTPUT_DIR, load_config  # noqa: E402

SCFG = load_config("serving")
ACFG = SCFG["api"]
_MISSING = SCFG["scoring"]["missing_feature_default"]
_TOP_N = ACFG["shap_top_factors"]

B = joblib.load(OUTPUT_DIR / "lgbm_model.pkl")
MODEL, FEATURES = B["model"], B["features"]
THR = B["threshold"]
DEC_EDGES = B.get("decile_edges")
P_GRID, VOL_GRID, RANK_Q = B.get("p_grid"), B.get("vol_grid"), B.get("rank_q")
VOLF = B.get("vol_feature", FEATURES[0])
W = B.get("priority_weights", {"probability": 0.70, "volume": 0.30})
N_DEC = SCFG["decile_tiering"]["n_deciles"]

app = FastAPI(title=ACFG["title"], version=str(ACFG["version"]))

# A worked example body: a plausible port on the service feature, 0 elsewhere.
# Derived from the bundle's feature list, so it adapts if the feature set changes.
_PORT_FEATURE = next((f for f in FEATURES if "DST_PORT" in f.upper()
                      or "DESTINATION PORT" in f.upper()), None)
EXAMPLE = {f: (22 if f == _PORT_FEATURE else 0.0) for f in FEATURES}


def _vec(body: Dict[str, Any]) -> dict:
    row = {f: float(body.get(f, _MISSING) or _MISSING) for f in FEATURES}
    row["_vol"] = float(np.log1p(float(body.get(VOLF, 0) or 0)))
    return row


def _pct(v, grid):
    """Map a raw value onto its training-set percentile via the stored grid."""
    if not grid:
        return np.zeros_like(np.asarray(v, dtype=float))
    return np.interp(np.asarray(v, dtype=float), np.asarray(grid), np.asarray(RANK_Q))


def _score(rows):
    vol = np.array([r.pop("_vol") for r in rows], dtype=float)
    X = (pd.DataFrame(rows)[FEATURES]
         .apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32))
    p = MODEL.predict_proba(X)[:, 1]
    pri = 100.0 * (W["probability"] * _pct(p, P_GRID) + W["volume"] * _pct(vol, VOL_GRID))
    dec = (np.clip(np.searchsorted(np.array(DEC_EDGES[1:-1]), pri, side="right") + 1, 1, N_DEC)
           if DEC_EDGES else np.clip((pri / (100 / N_DEC)).astype(int) + 1, 1, N_DEC))
    return p, pri, dec, X


def _tier(decile: int) -> str:
    actions = SCFG["decile_tiering"]["actions"]
    return actions.get(str(decile), SCFG["decile_tiering"]["default_action"])


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM — CSE-CIC-IDS2018 native-only",
            "n_features": len(FEATURES), "threshold_alert": round(THR, 6),
            "operating_point": "highest recall at >=70% precision"}


@app.get("/features")
def features():
    return {"request_body_accepts_NATIVE_only": FEATURES, "derived": [],
            "note": f"POST /score a JSON object of these native features. "
                    f"Missing -> {_MISSING}."}


@app.get("/example")
def example():
    return EXAMPLE


@app.get("/metrics")
def metrics():
    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/score")
def score(body: Dict[str, Any] = Body(..., examples=[EXAMPLE])):
    try:
        p, pri, dec, X = _score([_vec(body)])
    except Exception as e:                                    # noqa: BLE001
        raise HTTPException(422, f"scoring failed: {e}")
    sv = MODEL.booster_.predict(X, pred_contrib=True)[0][:-1]
    top = pd.Series(sv, index=FEATURES).sort_values(key=abs, ascending=False).head(_TOP_N)
    verdict = "ALERT" if p[0] >= THR else "NORMAL"
    is_alert = verdict == "ALERT"
    return {"probability": round(float(p[0]), 6), "verdict": verdict,
            "priority": round(float(pri[0]), 2) if is_alert else None,
            "decile": int(dec[0]) if is_alert else None,
            "action": _tier(int(dec[0])) if is_alert else SCFG["routing"]["noise_destination"],
            "decile_basis": "alerts only — normal flows are not ranked",
            "top_factors": [{"feature": k, "shap": round(float(v), 4),
                             "pushes": "ALERT" if v > 0 else "NORMAL"} for k, v in top.items()]}


@app.post("/score/batch")
def score_batch(flows: List[Dict[str, Any]] = Body(...)):
    if not flows:
        raise HTTPException(422, "empty body")
    p, pri, dec, _ = _score([_vec(f) for f in flows])
    out = [{"index": i, "probability": round(float(p[i]), 6),
            "verdict": "ALERT" if p[i] >= THR else "NORMAL",
            "priority": round(float(pri[i]), 2) if p[i] >= THR else None,
            "decile": int(dec[i]) if p[i] >= THR else None} for i in range(len(p))]
    alerts = sorted([r for r in out if r["verdict"] == "ALERT"],
                    key=lambda r: -(r["priority"] or 0))
    normal = [r for r in out if r["verdict"] == "NORMAL"]
    return {"count": len(out), "alerts": len(alerts), "ranked": alerts + normal}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ACFG["bind_host"], port=ACFG["port"])
