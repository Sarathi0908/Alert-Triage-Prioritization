"""Tests for src/evaluation/attack_class_evaluator.py.

Queue metrics and per-class recall on data where the right answer is arithmetic,
so a regression in the metric code cannot hide behind a plausible-looking number.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import attack_class_evaluator as ace


def test_precision_at_k_counts_the_top_k_only():
    y = [1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert ace.precision_at_k(y, scores, 2) == 1.0
    assert ace.precision_at_k(y, scores, 4) == 0.5


def test_ndcg_is_1_for_a_perfect_ranking():
    y = [1, 1, 0, 0]
    assert ace.ndcg_at_k(y, [0.9, 0.8, 0.2, 0.1], 4) == pytest.approx(1.0)


def test_ndcg_drops_for_an_inverted_ranking():
    y = [1, 1, 0, 0]
    good = ace.ndcg_at_k(y, [0.9, 0.8, 0.2, 0.1], 4)
    bad = ace.ndcg_at_k(y, [0.1, 0.2, 0.8, 0.9], 4)
    assert bad < good


def test_ndcg_of_all_negatives_is_zero_not_a_crash():
    assert ace.ndcg_at_k([0, 0, 0], [0.9, 0.5, 0.1], 3) == 0.0


def test_evaluate_flows_confusion_cells_and_fn_rate():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.05, 0.7, 0.05])
    pred = (scores >= 0.5).astype(int)      # tp=2 fn=1 fp=1 tn=4
    m = ace.evaluate_flows(y, scores, pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["fn_rate_%"] == pytest.approx(100 * 1 / 3, abs=1e-3)
    assert m["base_rate_%"] == pytest.approx(37.5)


def test_evaluate_flows_reports_the_overfit_gap_when_train_is_given():
    y = np.array([0, 0, 1, 1])
    te_scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = ace.evaluate_flows(y, te_scores, (te_scores >= 0.5).astype(int),
                           train_scores=te_scores, y_train=y)
    assert m["roc_gap"] == pytest.approx(0.0)
    assert "train_roc" in m


def test_alert_reduction_is_the_share_not_queued():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.1])
    m = ace.evaluate_flows(y, scores, (scores >= 0.5).astype(int))
    assert m["alert_reduction_%"] == pytest.approx(75.0)


def _test_frame():
    return pd.DataFrame({
        "attack_class": ["BENIGN", "BENIGN", "DoS", "DoS", "DoS", "Bot"],
        "y_is_attack": [0, 0, 1, 1, 1, 1],
    })


def test_per_class_recall_excludes_benign_and_sorts_by_volume():
    pred = [0, 0, 1, 1, 0, 0]            # DoS 2/3, Bot 0/1
    rows = ace.evaluate_attack_classes(_test_frame(), pred, "attack_class")
    assert [r["attack_class"] for r in rows] == ["DoS", "Bot"]
    # recall is stored rounded to 4dp, so compare at that resolution
    assert rows[0]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert rows[1]["recall"] == 0.0
    assert all(r["attack_class"].upper() != "BENIGN" for r in rows)


def test_weakest_class_identifies_the_blind_spot():
    pred = [0, 0, 1, 1, 1, 0]            # DoS 1.0, Bot 0.0
    rows = ace.evaluate_attack_classes(_test_frame(), pred, "attack_class")
    assert ace.weakest_class(rows)["attack_class"] == "Bot"


def test_weakest_class_of_nothing_is_none():
    assert ace.weakest_class([]) is None
