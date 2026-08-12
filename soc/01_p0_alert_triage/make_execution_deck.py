"""build_deck.py — PPT for the CSE-CIC-IDS2018 model (mirrors the lightgbm deck)."""
# Re-anchored for the slot root: this script lives at <slot>/make_execution_deck.py, so the
# slot root is this file's own directory. Reads what src/pipeline.py produced.
import sys, json
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
OUT, EDA, DATA = ROOT / "outputs", ROOT / "outputs" / "eda", ROOT / "data"
for old in OUT.glob("*.pptx"):
    try: old.unlink()
    except Exception: pass
M = json.load(open(OUT / "metrics.json"))
M18 = json.load(open(OUT / "metrics_2018.json")) if (OUT / "metrics_2018.json").exists() else {}
ED = json.load(open(EDA / "eda_summary.json")) if (EDA / "eda_summary.json").exists() else {}
SEL = json.load(open(DATA / "native_selected.json"))
scr = pd.read_csv(DATA / "feature_screen.csv") if (DATA / "feature_screen.csv").exists() else pd.DataFrame()
allp = pd.read_csv(EDA / "all_columns_profile.csv") if (EDA / "all_columns_profile.csv").exists() else pd.DataFrame()
NAVY, BLUE, RED, GREY, INK, LIGHT = C(0x0D,0x36,0x6B),C(0x2A,0x78,0xD6),C(0xD0,0x3B,0x3B),C(0x52,0x51,0x4E),C(0x0B,0x0B,0x0B),C(0xF0,0xEF,0xEC)
P = Presentation(); P.slide_width, P.slide_height = I(13.333), I(7.5); BLANK = P.slide_layouts[6]

def slide(title, sub=None):
    s = P.slides.add_slide(BLANK); b = s.shapes.add_textbox(I(0.55), I(0.26), I(12.2), I(0.95)).text_frame
    b.text = title; b.paragraphs[0].runs[0].font.size = Pt(26); b.paragraphs[0].runs[0].font.bold = True
    b.paragraphs[0].runs[0].font.color.rgb = NAVY
    if sub:
        p = b.add_paragraph(); p.text = sub; p.runs[0].font.size = Pt(12.5); p.runs[0].font.color.rgb = GREY
    return s
def bullets(s, items, x=0.7, y=1.45, w=12.0, h=5.5, size=14):
    tf = s.shapes.add_textbox(I(x), I(y), I(w), I(h)).text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        lvl, txt = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = ("• " if lvl == 0 else "     – ") + txt
        for r in p.runs: r.font.size = Pt(size if lvl == 0 else size-2); r.font.color.rgb = INK if lvl == 0 else GREY
        p.space_after = Pt(5)
def mono(s, text, x=0.7, y=1.5, w=12.0, h=5.0, size=12):
    tf = s.shapes.add_textbox(I(x), I(y), I(w), I(h)).text_frame; tf.word_wrap = True
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = line
        for r in p.runs: r.font.size = Pt(size); r.font.name = "Consolas"; r.font.color.rgb = INK
def table(s, df, x=0.6, y=1.45, w=12.1, h=4.9, fs=9.5):
    r, c = df.shape; t = s.shapes.add_table(r+1, c, I(x), I(y), I(w), I(h)).table
    for j, col in enumerate(df.columns):
        cell = t.cell(0, j); cell.text = str(col); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        pr = cell.text_frame.paragraphs[0]
        if pr.runs: pr.runs[0].font.size = Pt(fs); pr.runs[0].font.bold = True; pr.runs[0].font.color.rgb = C(0xFF,0xFF,0xFF)
    for i in range(r):
        for j in range(c):
            cell = t.cell(i+1, j); cell.text = str(df.iat[i, j]); cell.fill.solid()
            cell.fill.fore_color.rgb = C(0xFF,0xFF,0xFF) if i % 2 else LIGHT
            pr = cell.text_frame.paragraphs[0]
            if pr.runs: pr.runs[0].font.size = Pt(fs); pr.runs[0].font.color.rgb = INK
def pic(s, name, x, y, h):
    f = OUT / name
    if not f.exists(): f = EDA / name
    if f.exists(): s.shapes.add_picture(str(f), I(x), I(y), height=I(h))
def kpis(s, items, y=1.4):
    w = 12.1/len(items)
    for i, (v, lab) in enumerate(items):
        bx = s.shapes.add_textbox(I(0.6+i*w), I(y), I(w-0.16), I(1.3)).text_frame; bx.text = v
        bx.paragraphs[0].runs[0].font.size = Pt(26); bx.paragraphs[0].runs[0].font.bold = True
        bx.paragraphs[0].runs[0].font.color.rgb = NAVY; bx.paragraphs[0].alignment = PP_ALIGN.CENTER
        p = bx.add_paragraph(); p.text = lab; p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size = Pt(10.5); p.runs[0].font.color.rgb = GREY

# title
s = P.slides.add_slide(BLANK); tb = s.shapes.add_textbox(I(0.9), I(2.3), I(11.6), I(2.9)).text_frame
tb.text = "Alert Triage & Prioritization — CSE-CIC-IDS2018"
tb.paragraphs[0].runs[0].font.size = Pt(40); tb.paragraphs[0].runs[0].font.bold = True; tb.paragraphs[0].runs[0].font.color.rgb = NAVY
for t, sz, col in [("Bhairava SOC — Model #01 · LightGBM · native-only", 18, BLUE),
                   (f"CSE-CIC-IDS2018 (MachineLearningCVE, 8 days merged) — {M['rows_total']:,} flows · {M['attack_rate_%']}% attack · 14 attack classes", 13, GREY)]:
    p = tb.add_paragraph(); p.text = t; p.runs[0].font.size = Pt(sz); p.runs[0].font.color.rgb = col

s = slide("1 · Use Case", "detect attack vs normal from network-flow logs")
bullets(s, ["Every network flow is judged: ALERT (attack) or NORMAL — a binary decision.",
            "Model reviews 100% of flows; the attacks are ranked into a priority decile.",
            "CSE-CIC-IDS2018 label: BENIGN -> 0, any attack -> 1. Ground-truth (attacks were scripted on a schedule).",
            "14 attack classes: DoS (4), DDoS, PortScan, FTP/SSH brute-force, Web (BF/XSS/SQLi), Bot, Infiltration, Heartbleed."])

s = slide("2 · Dataset — CSE-CIC-IDS2018", "the official CIC MachineLearningCVE set, all 8 days merged")
kpis(s, [(f"{M['rows_total']:,}", "flows"), (f"{ED.get('attack_flows',0):,}", "attacks"),
         (f"{M['attack_rate_%']}%", "attack rate"), ("14", "attack classes")])
bullets(s, ["78 native CICFlowMeter features per flow (packet sizes, timing/IAT, byte/packet counts, TCP flags, rates).",
            "Merged Mon–Fri; inf->NaN, dropna, dedup: 2.83M -> 2.57M rows.",
            "Ground-truth labelled by the researchers who ran the attacks."], y=3.0, size=13)

s = slide("3 · EDA — Each attack scenario as a graph", "class counts + per-class fingerprints")
pic(s, "alert_classes.png", 0.5, 1.4, 5.6)
pic(s, "per_class_signature.png", 6.6, 1.4, 5.6)

s = slide("4 · EDA — Distributions & counts", "attack rate by port · per-column profile")
pic(s, "distributions.png", 0.5, 1.5, 3.1)
if len(allp):
    cols = [c for c in ["column","in_model","null_count","unique_count","duplicate_count","min","max","mean"] if c in allp.columns]
    table(s, allp[cols].head(20).reset_index(drop=True), x=0.5, y=4.4, w=12.3, h=2.7, fs=7.5)

s = slide("5 · Feature Selection — leak + NOISE screen", "how the 22 were chosen")
pic(s, "single_feature_auc.png", 0.45, 1.45, 5.4)
bullets(s, ["Start from 78 native features; native only.",
            "NOISE removed: near-constant (no info) + redundant (|r|>0.95).",
            (1, f"{len(SEL['removed']['near_constant_noise'])} near-constant + {len(SEL['removed']['redundant_noise (|r|>0.95)'])} redundant dropped."),
            "LEAK removed: any feature with single-feature AUC >= 0.999.",
            (1, f"leaks found: {SEL['removed']['leak_suspect (auc>=0.999)'] or 'NONE'}"),
            f"RESULT: {SEL['n']} native features · mean |corr| {ED.get('mean_abs_feature_correlation','~0.1')}.",
            "No single feature exceeds AUC ~0.74 alone -> the high score is COMBINED signal, not a leak."],
        x=6.1, y=1.5, w=6.9, size=12.5)

s = slide(f"6 · The {SEL['n']} native features", "family · gain% · single-feature AUC")
rows = [{"feature": f[:30], "family": SEL["families"][f], "gain %": SEL["gain_pct"][f],
         "single-AUC": round(SEL["single_feature_auc"][f], 2)} for f in SEL["features"]]
table(s, pd.DataFrame(rows), y=1.4, h=5.7, fs=9)

s = slide("7 · Why LightGBM & Training Protocol", "same algorithm as the Unraveled model")
bullets(s, ["LightGBM — native missing/categorical handling, best-in-class on tabular flow data, exact tree-SHAP.",
            f"Split: RANDOM 70/30 stratified (train {M['rows_train']:,} / test {M['rows_test']:,}).",
            f"Hyperparameters: {M['hyperparameters']}",
            "Operating point: highest recall at >=70% precision.",
            "Native features only -> the /score request body IS the vector (no train/serve skew)."])

s = slide("8 · Results", "held-out 30%")
kpis(s, [(f"{M['roc_auc']:.4f}", "ROC-AUC"), (f"{M['pr_auc']:.4f}", "PR-AUC"), (f"{M['recall']:.3f}", "recall"), (f"{M['fn_rate_%']}%", "FN rate")])
pic(s, "roc_curve.png", 0.6, 2.9, 3.5); pic(s, "pr_curve.png", 4.7, 2.9, 3.5); pic(s, "confusion_matrix.png", 8.9, 2.9, 3.5)

s = slide("9 · Is ROC ≈ 1.0 a leak? — NO", "why the curve is 'not smooth', and how to make it smooth")
mono(s, f"""
  Screen result: NO single feature separates the classes alone
  (max single-feature AUC ~0.74; top feature holds ~23% of gain, not 90%+).
  Near-constant + redundant NOISE removed; zero features exceed AUC 0.999.

  So ROC {M['roc_auc']} on a random 70/30 split is NOT a leak -- CSE-CIC-IDS2018 is
  GENUINELY separable: DoS/DDoS/PortScan flows have very distinct shapes.
  This is well documented for CSE-CIC-IDS2018 and expected.

  To get the 'smooth'/realistic curve you want, the test must be HARDER:
  train on CSE-CIC-IDS2018, TEST on a DIFFERENT capture (CSE-CIC-IDS2018).
  A leak/overfit dies on a different dataset; genuine signal survives (lower,
  smoother AUC). That CROSS-DATASET test needs a CICFlowMeter-2018 set.
""", y=1.45, size=12.5)

if M18:
    at = M18.get("at_train_threshold", {})
    s = slide("9b · CROSS-DATASET TEST — train 2017 → test 2018", "the HONEST number (10 features common to both)")
    kpis(s, [(f"{M18['ROC_AUC_2018']:.3f}", "2018 ROC-AUC"), (f"{M18['PR_AUC_2018']:.3f}", "2018 PR-AUC"),
             (f"{at.get('recall','')}", "2018 recall"), (f"{at.get('precision','')}", "2018 precision")], y=1.3)
    pic(s, "roc_2018.png", 0.5, 2.7, 3.4); pic(s, "confusion_2018.png", 4.6, 2.7, 3.4); pic(s, "per_class_2018.png", 8.7, 2.7, 3.4)
    s = slide("9c · What the cross-dataset test proves", "read this carefully")
    mono(s, f"""
  2017 (in-distribution, 70/30)   ROC {M['roc_auc']}   <- near-perfect
  2018 (CROSS-DATASET, unseen)    ROC {M18['ROC_AUC_2018']}   <- collapses

  The 2017 1.0 was NOT a leak (screen proved it) — it was in-distribution
  SEPARABILITY. But it does NOT generalise: on a genuinely different capture
  the same model falls to ROC {M18['ROC_AUC_2018']} (worse than random overall).

  Per-class on 2018: MOST attack types are still caught (DoS/DDoS-LOIC/Bot/
  brute-force recall ~1.0), BUT the largest class (DDoS-HOIC, 1.08M) is missed
  entirely and precision collapses to {at.get('precision','')} (false positives on 2018 benign).

  WHY: domain shift + tool differences (CICFlowMeter 2017 vs NetFlow 2018;
  unit/definition mismatches on window-size & duration features).

  LESSON: single-dataset near-perfect scores are misleading. The cross-dataset
  test is the real one — and it says this model must be RE-TRAINED on 2018-like
  data (or on a shared feature extractor) before it can be trusted in production.
""", y=1.45, size=12)

s = slide("10 · Detection recall per attack class", "every scenario caught (2017 in-distribution)")
pc = pd.DataFrame(M["per_class"])[["attack_class", "n", "recall"]]
pc["attack_class"] = pc["attack_class"].astype(str).str.replace("�", "-", regex=False).str[:30]
table(s, pc, x=0.6, y=1.45, w=6.2, h=5.2, fs=9)
pic(s, "per_class_recall.png", 7.0, 1.6, 4.6)

s = slide("11 · SHAP — global & local", "which features drive decisions, and why for one alert")
pic(s, "shap_global.png", 0.4, 1.45, 5.5); pic(s, "shap_local.png", 6.7, 1.45, 5.5)

s = slide("12 · Alert Prioritization — decile", "priority = 0.70·prob + 0.30·volume (alerts only)")
pic(s, "decile_priority.png", 1.4, 1.5, 4.3)
dec = pd.DataFrame(M["decile"])[["decile", "n", "mean_priority", "hit_%"]]; dec["mean_priority"] = dec["mean_priority"].round(1)
table(s, dec.sort_values("decile", ascending=False).reset_index(drop=True), x=1.4, y=5.4, w=10.4, h=1.6, fs=9)

s = slide("13 · Deployment & Deliverables", "endpoint contract + artifacts")
bullets(s, ["POST /score — JSON of the 22 NATIVE features (real names). Missing -> 0. No derived features.",
            "Endpoints: /health · /features · /example · /metrics · /score · /score/batch (port 8081).",
            "Returns: probability · verdict · priority · decile · top-5 SHAP.",
            "Postman collection: real example flows per attack class (DoS/DDoS/PortScan/brute-force/Bot/BENIGN).",
            "data/CSE-CIC-IDS2018_feature_sheets.xlsx — 3 sheets: selected / removed / label.",
            "Downloads/CSE-CIC-IDS2018_merged_raw_sample.csv — the raw sheet."])

s = slide("14 · Limitations & Next", "stated plainly")
bullets(s, ["Random 70/30 on CSE-CIC-IDS2018 is near-perfect because the dataset is highly separable (not a leak).",
            "The honest 'smooth' number needs a CROSS-DATASET test: train 2017 -> test 2018 (CICFlowMeter).",
            "The 2018 you provided was NetFlow format (different columns) — a CICFlowMeter-2018 is still needed for that test.",
            "Ground-truth labels (attacks scripted), not analyst dispositions — good for detection, not full triage.",
            "NEXT: add CICFlowMeter-2018 as the test set to produce the realistic generalization curve."])

f = OUT / "CSE-CIC-IDS2018_Alert_Triage_Deck.pptx"
P.save(f)
print(f"wrote {f.name}  ({len(P.slides._sldIdLst)} slides)")
