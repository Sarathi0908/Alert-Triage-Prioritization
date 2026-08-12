"""
build_notebooks.py  —  generate the five Alert Triage notebook *sources* (outputs
stripped) that document the workflow encoded in src/. Notebooks are for
exploration/reporting; the reusable logic lives in src/ (instruction §3-§4).

Run:  py notebooks/build_notebooks.py
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from pathlib import Path

HERE = Path(__file__).resolve().parent

_BOOT = """import sys
from pathlib import Path
SLOT = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(SLOT))
from src.config import load_all_configs
from src import data_source as ds
cfg = load_all_configs()
mcfg, fcfg = cfg['model'], cfg['feature']
selected = ds.load_selected(mcfg)['features']
print('dataset', mcfg['data']['dataset'], '| model features', selected)
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def save(name, cells):
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, str(HERE / name))
    print("wrote", name)


# 01 — EDA / flow-traffic profile --------------------------------------------
save("01_eda_flow_traffic_profile.ipynb", [
    md("# 01 · EDA — flow traffic profile (Alert Triage #01)\n"
       "Explore the merged NetFlow table: attack-class volumes, per-class feature "
       "fingerprints, and how separable the traffic is. Reusable logic lives in "
       "`src/evaluation/flow_traffic_profiler.py`; this notebook only explores."),
    code(_BOOT),
    md("## Load the table (all native columns)"),
    code("native = ds.native_features(mcfg)\n"
         "df = ds.load_frame(mcfg, columns=native)\n"
         "target, cls = mcfg['target'], mcfg['attack_class_column']\n"
         "print(f'rows={len(df):,}  attack={df[target].mean()*100:.3f}%  native={len(native)}')"),
    md("## Attack classes — every campaign in the capture"),
    code("vc = df[df[cls].str.upper() != 'BENIGN'][cls].value_counts()\n"
         "vc.plot.barh(figsize=(9,6), color='#d03b3b', logx=True,\n"
         "             title='Attack scenarios (count, log)');\n"
         "print(vc.to_string())"),
    md("## Column profile — counts, not percentages\n"
       "`null_count` · `unique_count` · `duplicate_count` (distinct values appearing >1)."),
    code("from src.evaluation import flow_traffic_profiler as ftp\n"
         "ftp.profile(df, selected + [target])"),
    md("## Class-level share of traffic"),
    code("(df[cls].value_counts(normalize=True) * 100).round(3).head(20)"),
])

# 02 — leak and noise screen -------------------------------------------------
save("02_leak_and_noise_screen.ipynb", [
    md("# 02 · Leak & noise screen (Alert Triage #01)\n"
       "Why the model keeps only a handful of native columns. NOISE = near-constant "
       "or redundant. LEAK = a column that alone almost perfectly separates the "
       "classes. Then a DOMINANCE screen keeps the gain balanced.\n\n"
       "Thresholds come from `configs/feature_config.yaml`; the logic is "
       "`src/features/native_feature_screener.py`."),
    code(_BOOT),
    md("## The screen's own audit trail"),
    code("import pandas as pd\n"
         "scr = pd.read_csv(SLOT/'data'/mcfg['data']['screen_audit'])\n"
         "scr[['feature','family','role','gain_pct','single_feature_auc','zero_frac']]"),
    md("## What was removed, and why"),
    code("sel = ds.load_selected(mcfg)\n"
         "for reason, cols in sel['removed'].items():\n"
         "    print(f'{reason}: {len(cols)}')\n"
         "    for c in cols: print('   ', c)"),
    md("## Single-feature AUC — the leak check\n"
       "No kept feature may separate the classes on its own; that would be a data "
       "flaw the model memorises rather than behaviour it learns."),
    code("floor = fcfg['screen']['leak']['single_feature_auc_min']\n"
         "ax = scr.set_index('feature').single_feature_auc.sort_values().plot.barh(\n"
         "        figsize=(8,8), color='#2a78d6')\n"
         "ax.axvline(floor, ls='--', c='#d03b3b', label=f'leak floor {floor}'); ax.legend();\n"
         "print('max single-feature AUC:', scr.single_feature_auc.max())"),
    md("## Gain balance — no dominator, no dead weight"),
    code("bal = fcfg['screen']['balance']\n"
         "g = sel['gain_pct']\n"
         "print('cap', bal['gain_cap_pct'], '| max kept', max(g.values()))\n"
         "print('floor', bal['gain_floor_pct'], '| min kept', min(g.values()))\n"
         "g"),
])

# 03 — LightGBM triage baseline ---------------------------------------------
save("03_lightgbm_triage_baseline.ipynb", [
    md("# 03 · LightGBM triage baseline (Alert Triage #01)\n"
       "Fit the classifier the pipeline fits, on the same split, and look at what it "
       "learned. The model class is `src/models/lightgbm_triage.LightGBMTriage`."),
    code(_BOOT),
    md("## Split and fit"),
    code("import numpy as np\n"
         "from sklearn.model_selection import train_test_split\n"
         "from src.models.lightgbm_triage import LightGBMTriage\n"
         "df = ds.load_frame(mcfg, columns=selected)\n"
         "y = df[mcfg['target']].astype(int).values\n"
         "tr, te = train_test_split(np.arange(len(df)), test_size=mcfg['split']['test_frac'],\n"
         "                          random_state=mcfg['seed'], stratify=y)\n"
         "model = LightGBMTriage(mcfg['lgbm_classifier'], mcfg['seed']).fit(\n"
         "            df.iloc[tr][selected], selected, y[tr])\n"
         "print('scale_pos_weight (from the split, not configured):', model.scale_pos_weight)"),
    md("## Score the held-out split"),
    code("p = model.attack_probability(df.iloc[te][selected])\n"
         "from src.evaluation import attack_class_evaluator as ace\n"
         "thr = model.fit_threshold(y[te], p, mcfg['operating_point']['min_precision'],\n"
         "                          mcfg['operating_point']['fallback_threshold'])\n"
         "print('fitted threshold:', round(thr, 6))\n"
         "ace.evaluate_flows(y[te], p, model.verdict(p), model.attack_probability(df.iloc[tr][selected]), y[tr])"),
    md("## Split-gain importance"),
    code("model.gain_importance().sort_values().plot.barh(figsize=(8,4), color='#2a78d6',\n"
         "        title='% of total gain');"),
    md("## Why a flow scored — exact tree SHAP from the booster itself"),
    code("import pandas as pd\n"
         "Xs = df.iloc[te][selected].head(2000)\n"
         "sv = model.shap_contributions(Xs)\n"
         "ps = model.attack_probability(Xs); i = int(ps.argmax())\n"
         "print('explaining the highest-scoring flow, p =', round(float(ps[i]), 6))\n"
         "pd.Series(sv[i], index=selected).sort_values()"),
])

# 04 — operating point and priority -----------------------------------------
save("04_operating_point_and_priority.ipynb", [
    md("# 04 · Operating point & priority (Alert Triage #01)\n"
       "The threshold is *fitted*, not fixed at 0.5: the analyst queue is "
       "precision-constrained, so we hold a precision floor and maximise recall "
       "under it. Priority then orders the queue — over PREDICTED ALERTS ONLY.\n\n"
       "Logic: `src/features/priority_score_builder.PriorityScoreBuilder`."),
    code(_BOOT),
    md("## The precision/recall trade-off the threshold is chosen from"),
    code("import json\n"
         "m = json.load(open(SLOT/'outputs'/'metrics.json'))\n"
         "print('floor      ', m['min_precision_floor'])\n"
         "print('threshold  ', m['threshold'])\n"
         "print('precision  ', m['precision'], '| recall', m['recall'])\n"
         "print('FN rate    ', m['fn_rate_%'], '%')\n"
         "print('reduction  ', m['alert_reduction_%'], '% of flows suppressed as NORMAL')"),
    md("## Priority deciles — mean priority and hit rate per decile"),
    code("import pandas as pd\n"
         "pd.DataFrame(m['decile'])"),
    md("## What the priority formula is"),
    code("print(m['decile_basis']['formula'])\n"
         "print(m['decile_basis']['note'])\n"
         "print('alerts ranked:', m['decile_basis']['n_alerts'])"),
    md("## Routing — what the backend does with each verdict"),
    code("print(json.dumps(m['routing'], indent=2))"),
])

# 05 — attack-class evaluation ----------------------------------------------
save("05_attack_class_evaluation.ipynb", [
    md("# 05 · Attack-class evaluation & acceptance gates (Alert Triage #01)\n"
       "An aggregate ROC of 0.99 can still hide an attack class the model never "
       "catches, so recall is checked per class. Then the plan §8 gates are applied.\n\n"
       "Logic: `src/evaluation/attack_class_evaluator.py` and "
       "`src/evaluation/operating_point_validator.py`."),
    code(_BOOT),
    md("## Recall per attack class — the blind-spot check"),
    code("import json, pandas as pd\n"
         "m = json.load(open(SLOT/'outputs'/'metrics.json'))\n"
         "pc = pd.DataFrame(m['per_class']).sort_values('recall')\n"
         "pc.plot.barh(x='attack_class', y='recall', figsize=(9,5), color='#2a78d6',\n"
         "             title='Detection recall per attack class', legend=False);\n"
         "pc"),
    md("## The weakest class is what drift monitoring watches"),
    code("print(m['weakest_attack_class'])"),
    md("## Acceptance gates (plan §8)"),
    code("g = m['acceptance']\n"
         "for k, v in g.items():\n"
         "    print(f'{k:34s} {v}')"),
    md("## Queue quality — analysts work the top of the ranking"),
    code("print('NDCG@100        ', m['ndcg_at_100'])\n"
         "print('precision@1000  ', m['precision_at_1000'])\n"
         "print('lift@1000       ', m['lift_at_1000'], 'x random')"),
])
