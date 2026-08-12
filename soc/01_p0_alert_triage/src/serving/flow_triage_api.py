"""
flow_triage_api.py  —  REST scoring endpoint for the Alert Triage model (plan §6).

REQUEST BODY = a JSON object of the model's NATIVE NetFlow features, by their real
names (e.g. "L4_DST_PORT", "SRC_TO_DST_AVG_THROUGHPUT"). No derived features — the
body IS the feature vector, so the contract cannot drift from what the model
expects. Missing features default to the configured value.

The feature list, the alert threshold and the priority percentile grids all come
from the model bundle, so the served operating point is the evaluated one.

Run:  py src/serving/flow_triage_api.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import OUTPUT_DIR, load_config                          # noqa: E402
from src.features.priority_score_builder import PriorityScoreBuilder    # noqa: E402
from src.models.lightgbm_triage import LightGBMTriage                   # noqa: E402
from src.serving import siem_platform_connector as sc                   # noqa: E402

SCFG = load_config("serving")
ACFG = SCFG["api"]
_MISSING = SCFG["scoring"]["missing_feature_default"]
_TOP_N = ACFG["shap_top_factors"]

MODEL, BUNDLE = LightGBMTriage.load(str(OUTPUT_DIR / "lgbm_model.pkl"))
PRIORITY = PriorityScoreBuilder.from_bundle(BUNDLE)
FEATURES = MODEL.feature_names
VOLF = PRIORITY.volume_feature

app = FastAPI(title=ACFG["title"], version=str(ACFG["version"]))

# A worked example: a plausible port on the service feature, 0 elsewhere. Derived
# from the bundle's feature list, so it adapts if the feature set changes.
_PORT_FEATURE = next((f for f in FEATURES if "DST_PORT" in f.upper()
                      or "DESTINATION PORT" in f.upper()), None)
EXAMPLE = {f: (22 if f == _PORT_FEATURE else 0.0) for f in FEATURES}


def _vector(body: Dict[str, Any]) -> dict:
    """Body -> one feature row, plus the raw volume term the priority blend needs."""
    row = {f: float(body.get(f, _MISSING) or _MISSING) for f in FEATURES}
    row["_vol"] = float(body.get(VOLF, 0) or 0)
    return row


def _score(rows: list[dict]):
    raw_vol = np.array([r.pop("_vol") for r in rows], dtype=float)
    X = (pd.DataFrame(rows)[FEATURES]
         .apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32))
    p = MODEL.attack_probability(X)
    vol = np.log1p(raw_vol) if PRIORITY.log1p_volume else raw_vol
    pri = PRIORITY.transform(p, vol)
    return p, pri, PRIORITY.decile(pri), X


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM — CSE-CIC-IDS2018 native-only",
            "n_features": len(FEATURES), "threshold_alert": round(MODEL.threshold, 6),
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
        p, pri, dec, X = _score([_vector(body)])
    except Exception as e:                                    # noqa: BLE001
        raise HTTPException(422, f"scoring failed: {e}")
    contrib = MODEL.shap_contributions(X)[0]
    top = pd.Series(contrib, index=FEATURES).sort_values(key=abs, ascending=False).head(_TOP_N)
    verdict = "ALERT" if p[0] >= MODEL.threshold else "NORMAL"
    is_alert = verdict == "ALERT"
    record = sc.to_risk_record(0, float(p[0]), verdict,
                               float(pri[0]) if is_alert else None,
                               int(dec[0]) if is_alert else None, SCFG)
    return {"probability": round(float(p[0]), 6), "verdict": verdict,
            "priority": round(float(pri[0]), 2) if is_alert else None,
            "decile": int(dec[0]) if is_alert else None,
            "action": record["action"], "destination": record["destination"],
            "escalate": record["escalate"],
            "decile_basis": "alerts only — normal flows are not ranked",
            "top_factors": [{"feature": k, "shap": round(float(v), 4),
                             "pushes": "ALERT" if v > 0 else "NORMAL"}
                            for k, v in top.items()]}


@app.post("/score/batch")
def score_batch(flows: List[Dict[str, Any]] = Body(...)):
    if not flows:
        raise HTTPException(422, "empty body")
    p, pri, dec, _ = _score([_vector(f) for f in flows])
    out = [{"index": i, "probability": round(float(p[i]), 6),
            "verdict": "ALERT" if p[i] >= MODEL.threshold else "NORMAL",
            "priority": round(float(pri[i]), 2) if p[i] >= MODEL.threshold else None,
            "decile": int(dec[i]) if p[i] >= MODEL.threshold else None}
           for i in range(len(p))]
    alerts = sorted([r for r in out if r["verdict"] == "ALERT"],
                    key=lambda r: -(r["priority"] or 0))
    normal = [r for r in out if r["verdict"] == "NORMAL"]
    return {"count": len(out), "alerts": len(alerts), "ranked": alerts + normal}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ACFG["bind_host"], port=ACFG["port"])
