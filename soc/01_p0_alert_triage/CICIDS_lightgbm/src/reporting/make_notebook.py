"""make_notebook.py — generate the EDA notebook under notebooks/.

Run:  py src/reporting/make_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SLOT_ROOT = Path(__file__).resolve().parents[2]
if str(_SLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLOT_ROOT))

from src.config import SLOT_ROOT, load_config  # noqa: E402
from src import data_source as ds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

mcfg = load_config("model")
NB = SLOT_ROOT / "notebooks"
NB.mkdir(exist_ok=True)
FEATURES = ds.load_selected(mcfg)["features"]
DATASET = mcfg["data"]["dataset"]
TARGET = mcfg["target"]
CLS = mcfg["attack_class_column"]
NB_NAME = "01_eda_cicids2017.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.splitlines(keepends=True)}


cells = [
md(f"""# EDA — CSE-CIC-IDS2018 · Alert Triage · LightGBM (native features only)

**Dataset:** CSE-CIC-IDS2018 (NF-v2 NetFlow), merged and cleaned by `data/build_ids2018_dataset.py`.
**Label:** `BENIGN -> 0`, any attack `-> 1`; the attack-class name is kept for per-scenario graphs.
**Rule:** the model uses only NATIVE NetFlow features (no derived), leak/noise-screened
by `src/features/feature_screener.py`.

Counts, not percentages: null_count · unique_count · duplicate_count (distinct values appearing >1)."""),
code(f"""import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA, EDA = ROOT/"data", ROOT/"outputs"/"eda"
DATASET = {DATASET!r}
TARGET, CLS = {TARGET!r}, {CLS!r}
FEATURES = {json.dumps(FEATURES)}
print("ready")"""),
md("## 1 · Load the merged NetFlow table"),
code("""df = pd.read_parquet(DATA/DATASET)
for c in FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
df = df.fillna(0); df[CLS] = df[CLS].astype(str).str.replace("\\ufffd","-",regex=False).str.strip()
print(f"rows={len(df):,}  attack={df[TARGET].mean()*100:.3f}%")
print(df[CLS].value_counts())"""),
md("## 2 · Each attack scenario as a graph"),
code("""vc = df[df[CLS].str.upper()!="BENIGN"][CLS].value_counts()
ax = vc[::-1].plot.barh(figsize=(9,6), color="#d03b3b", logx=True); ax.set_title("Attack scenarios (count, log)"); plt.show()"""),
md("## 3 · Per-class fingerprint (z-scored mean of each model feature)"),
code("""mu = df.groupby(CLS)[FEATURES].mean()
z = ((mu - df[FEATURES].mean())/(df[FEATURES].std()+1e-9)).clip(-3,3)
fig,ax = plt.subplots(figsize=(14,6)); im=ax.imshow(z.values, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=6)
ax.set_yticks(range(len(z))); ax.set_yticklabels(z.index, fontsize=8); fig.colorbar(im); plt.title("Attack fingerprints"); plt.show()"""),
md("## 4 · Leak screen — single-feature ROC-AUC (none should be ~1.0)"),
code("""smp = df[FEATURES+[TARGET]].sample(min(400000,len(df)), random_state=42); ys=smp[TARGET].values
sep = pd.Series({c: max((a:=roc_auc_score(ys, smp[c].values)),1-a) for c in FEATURES}).sort_values()
sep.plot.barh(figsize=(8,7), color=["#d03b3b" if v>0.9 else "#2a78d6" for v in sep]); plt.axvline(0.5, ls="--", c="grey"); plt.title("Single-feature AUC"); plt.show()
print("max single-feature AUC:", round(sep.max(),3), "-> no leak" if sep.max()<0.999 else "-> LEAK")"""),
md("## 5 · Correlation of model features (low = each adds info)"),
code("""corr = smp[FEATURES].corr()
fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=6)
ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=6); fig.colorbar(im); plt.show()
off=corr.abs().values; print("mean |corr|:", round(off[np.triu_indices_from(off,1)].mean(),3))"""),
md("## 6 · X and Y given to the model"),
code("""X = df[FEATURES]; Y = df[TARGET]
print(f"X: {X.shape[0]:,} x {X.shape[1]} native features"); print(f"Y: {Y.sum():,} attack / {(Y==0).sum():,} normal")
display(pd.concat([X.head(6), Y.head(6)], axis=1))"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}}}
f = NB / NB_NAME
with open(f, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {f}  ({len(cells)} cells)")
