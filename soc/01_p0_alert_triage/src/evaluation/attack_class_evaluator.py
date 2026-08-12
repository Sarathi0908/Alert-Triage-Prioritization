"""
attack_class_evaluator.py  —  detection metrics for Alert Triage (#01), per the
plan's §5 protocol.

Two views, because an aggregate can hide a blind spot:
  * FLOW level      — the usual discrimination/threshold metrics over the held-out split.
  * ATTACK-CLASS    — recall for each attack class separately. A model can score
                      ROC 0.99 while missing an entire low-volume class; the queue
                      only helps if every campaign type is detectable.
Plus the QUEUE view (§2 learning-to-rank): analysts work the top of the queue, so
precision@K, NDCG@K and lift are what the ordering is judged on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BENIGN = "BENIGN"


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of the top-k highest-scoring flows that really are attacks."""
    return float(np.asarray(y_true)[np.argsort(-np.asarray(scores))[:k]].mean())


def ndcg_at_k(y_true, scores, k: int) -> float:
    """Ranking quality at the head of the queue (1.0 = perfect ordering)."""
    gains = np.asarray(y_true)[np.argsort(-np.asarray(scores))[:k]]
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(np.asarray(y_true))[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_flows(y_true, scores, pred, train_scores=None, y_train=None) -> dict:
    """Flow-level metrics at the chosen operating point."""
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    base = float(y_true.mean())
    roc = float(roc_auc_score(y_true, scores))
    out = {
        "roc_auc": round(roc, 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "base_rate_%": round(base * 100, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, scores)), 6),
        "log_loss": round(float(log_loss(y_true, scores, labels=[0, 1])), 6),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fn_rate_%": round(float(fn / max(fn + tp, 1) * 100), 3),
        "ndcg_at_100": round(ndcg_at_k(y_true, scores, 100), 4),
        "precision_at_1000": round(precision_at_k(y_true, scores, 1000), 4),
        "lift_at_1000": round(precision_at_k(y_true, scores, 1000) / max(base, 1e-9), 2),
        "alert_reduction_%": round(float((1 - pred.mean()) * 100), 2),
    }
    if train_scores is not None and y_train is not None:
        tr_roc = float(roc_auc_score(np.asarray(y_train).astype(int), train_scores))
        out["train_roc"] = round(tr_roc, 4)
        out["roc_gap"] = round(tr_roc - roc, 4)
    return out


def evaluate_attack_classes(test: pd.DataFrame, pred, attack_class_col: str) -> list[dict]:
    """Recall per attack class, BENIGN excluded (it has no recall to speak of)."""
    tt = test.assign(_pred=np.asarray(pred).astype(int))
    attacks = tt[tt[attack_class_col].str.upper() != BENIGN]
    rows = [{"attack_class": a, "n": int(len(d)),
             "recall": round(float((d._pred == 1).mean()), 4)}
            for a, d in attacks.groupby(attack_class_col)]
    return sorted(rows, key=lambda r: -r["n"])


def weakest_class(per_class: list[dict]) -> dict | None:
    """The class most likely to be a blind spot — what drift monitoring watches."""
    return min(per_class, key=lambda r: r["recall"]) if per_class else None
