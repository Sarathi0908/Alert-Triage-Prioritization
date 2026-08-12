"""
build_collection.py — Postman collection for the Alert Triage model.

Scenario request bodies are REAL example flows pulled from the merged NetFlow
table by attack class, carrying exactly the model's native features (by their real
names). Adapts automatically if the feature set changes: the feature list comes
from the trained bundle, the scenarios and the port from configs/serving_config.yaml.

Run:  py src/reporting/build_collection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import OUTPUT_DIR, SLOT_ROOT, load_config  # noqa: E402
from src import data_source as ds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

mcfg, scfg = load_config("model"), load_config("serving")
ACFG, PCFG = scfg["api"], scfg["postman"]
HOST, PORT = ACFG["host"], str(ACFG["port"])
BASE = f"http://{HOST}:{PORT}"

PM = SLOT_ROOT / "postman"
PM.mkdir(exist_ok=True)
for old in PM.glob("*.json"):
    old.unlink()

B = joblib.load(OUTPUT_DIR / "lgbm_model.pkl")
FEATURES = B["features"]
with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as fh:
    M = json.load(fh)

cls_col = mcfg["attack_class_column"]
df = ds.load_frame(mcfg, columns=FEATURES)


def body_for(cls: str) -> dict:
    sub = df[df[cls_col] == cls]
    if not len(sub):
        return {f: 0 for f in FEATURES}
    r = sub.iloc[0]
    return {f: (int(r[f]) if float(r[f]).is_integer() else round(float(r[f]), 4))
            for f in FEATURES}


BODIES = {c: body_for(c) for c in PCFG["scenarios"] if len(df[df[cls_col] == c])}


def req(name, method, path, body=None, tests=None):
    it = {"name": name,
          "request": {"method": method,
                      "header": ([{"key": "Content-Type", "value": "application/json"}]
                                 if body is not None else []),
                      "url": {"raw": BASE + path, "protocol": "http",
                              "host": HOST.split("."), "port": PORT,
                              "path": [s for s in path.split("/") if s]}}}
    if body is not None:
        it["request"]["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2),
                                 "options": {"raw": {"language": "json"}}}
    it["event"] = [{"listen": "test", "script": {"type": "text/javascript", "exec":
                   ["pm.test('status 200', () => pm.response.to.have.status(200));"]
                   + (tests or [])}}]
    return it


items = [
    req("1. Health", "GET", "/health",
        tests=["const r=pm.response.json();",
               f"pm.test('{M['n_features']} features', () => r.n_features === {M['n_features']});"]),
    req("2. Feature contract", "GET", "/features"),
    req("3. Example body", "GET", "/example"),
    req("4. Metrics", "GET", "/metrics"),
]
for i, (cls, body) in enumerate(BODIES.items(), 5):
    exp = "expect NORMAL" if cls.upper() == "BENIGN" else "expect ALERT"
    items.append(req(f"{i}. Score — {cls} ({exp})", "POST", "/score", body,
                     tests=["const r=pm.response.json();",
                            "pm.test('scored', () => r.verdict !== undefined);"]))
items.append(req(f"{len(items) + 1}. Batch score", "POST", "/score/batch",
                 list(BODIES.values()),
                 tests=["const r=pm.response.json();",
                        f"pm.test('{len(BODIES)} scored', () => r.count === {len(BODIES)});"]))

col = {"info": {"name": "Bhairava Model 01 — CSE-CIC-IDS2018 Alert Triage",
                "description": (
                    f"LightGBM on {M['dataset']} ({M['rows_total']:,} flows, "
                    f"{M['attack_rate_%']}% attack, {len(M.get('per_class', []))} attack classes). "
                    f"{M['n_features']} NATIVE NetFlow features (no derived).\n"
                    f"{M['protocol']}: ROC-AUC {M['roc_auc']} · PR-AUC {M['pr_auc']} · "
                    f"recall {M['recall']} · precision {M['precision']}.\n\n"
                    "REQUEST BODY = a JSON object of the native features (real names). "
                    "Missing -> 0.\nBodies below are REAL example flows per attack class.\n\n"
                    "Start server:  cd CICIDS_lightgbm && py src/serving/api.py"),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
       "item": items}

f = PM / PCFG["collection_name"]
with open(f, "w", encoding="utf-8") as fh:
    json.dump(col, fh, indent=2)
print(f"wrote {f.name}  ({len(items)} requests, real flows for: {list(BODIES.keys())})")
