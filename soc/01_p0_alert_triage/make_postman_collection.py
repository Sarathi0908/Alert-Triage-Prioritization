"""
build_collection.py — Postman collection for the CSE-CIC-IDS2018 model.

Scenario request bodies are REAL example flows pulled from the merged CSE-CIC-IDS2018 table by
attack class, carrying exactly the model's native features (by their real names). Adapts
automatically if the feature set changes.
"""
# Re-anchored for the slot root: this script lives at <slot>/make_postman_collection.py, so the
# slot root is this file's own directory. Reads what src/pipeline.py produced.
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
OUT, PM, DATA = ROOT / "outputs", ROOT / "postman", ROOT / "data"
PM.mkdir(exist_ok=True)
for old in PM.glob("*.json"): old.unlink()
BASE = "http://127.0.0.1:8081"

B = joblib.load(OUT / "lgbm_model.pkl")
FEATURES = B["features"]
M = json.load(open(OUT / "metrics.json"))
df = pd.read_parquet(DATA / "ids2018_merged.parquet", columns=FEATURES + ["attack_class"])
df["attack_class"] = df["attack_class"].astype(str).str.replace("�", "-", regex=False).str.strip()

SCEN = ["DoS attacks-Hulk", "DDOS attack-HOIC", "DDoS attacks-LOIC-HTTP", "SSH-Bruteforce",
        "Infilteration", "Bot", "Benign"]


def body_for(cls):
    sub = df[df.attack_class == cls]
    if not len(sub): return {f: 0 for f in FEATURES}
    r = sub.iloc[0]
    return {f: (int(r[f]) if float(r[f]).is_integer() else round(float(r[f]), 4)) for f in FEATURES}


BODIES = {c: body_for(c) for c in SCEN if len(df[df.attack_class == c])}


def req(name, method, path, body=None, tests=None):
    it = {"name": name, "request": {"method": method,
          "header": ([{"key": "Content-Type", "value": "application/json"}] if body is not None else []),
          "url": {"raw": BASE + path, "protocol": "http", "host": ["127", "0", "0", "1"], "port": "8081",
                  "path": [s for s in path.split("/") if s]}}}
    if body is not None:
        it["request"]["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2),
                                 "options": {"raw": {"language": "json"}}}
    it["event"] = [{"listen": "test", "script": {"type": "text/javascript", "exec":
                   ["pm.test('status 200', () => pm.response.to.have.status(200));"] + (tests or [])}}]
    return it


items = [
    req("1. Health", "GET", "/health", tests=["const r=pm.response.json();",
        f"pm.test('{M['n_features']} features', () => r.n_features === {M['n_features']});"]),
    req("2. Feature contract", "GET", "/features"),
    req("3. Example body", "GET", "/example"),
    req("4. Metrics", "GET", "/metrics"),
]
for i, (cls, body) in enumerate(BODIES.items(), 5):
    exp = "expect NORMAL" if cls.upper() == "BENIGN" else "expect ALERT"
    items.append(req(f"{i}. Score — {cls} ({exp})", "POST", "/score", body,
                     tests=["const r=pm.response.json();", "pm.test('scored', () => r.verdict !== undefined);"]))
items.append(req(f"{len(items)+1}. Batch score", "POST", "/score/batch", list(BODIES.values()),
                 tests=["const r=pm.response.json();", f"pm.test('{len(BODIES)} scored', () => r.count === {len(BODIES)});"]))

col = {"info": {"name": "Bhairava Model 01 — CSE-CIC-IDS2018 Alert Triage",
                "description": (f"LightGBM on CSE-CIC-IDS2018 ({M['rows_total']:,} flows, {M['attack_rate_%']}% attack, "
                                f"14 attack classes). {M['n_features']} NATIVE CICFlowMeter features (no derived).\n"
                                f"Random 70/30: ROC-AUC {M['roc_auc']} · PR-AUC {M['pr_auc']} · recall {M['recall']} "
                                f"· precision {M['precision']}.\n\n"
                                "REQUEST BODY = a JSON object of the native features (real names). Missing -> 0.\n"
                                "Bodies below are REAL example flows per attack class.\n\n"
                                "Start server:  cd soc/01_p0_alert_triage && py src/serving/flow_triage_api.py"),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
       "item": items}
f = PM / "bhairava_cse_cic_ids2018_alert_triage.postman_collection.json"
json.dump(col, open(f, "w", encoding="utf-8"), indent=2)
print(f"wrote {f.name}  ({len(items)} requests, real flows for: {list(BODIES.keys())})")
