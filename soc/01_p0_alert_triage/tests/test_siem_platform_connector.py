"""Tests for src/serving/siem_platform_connector.py.

The connector decides nothing about risk — it formats the model's decision for the
platform. What it must get right: unranked NORMAL flows carry no queue position
(and never NaN, which is not valid JSON), routing follows config, and the HITL rule
that a high-confidence flow is never auto-closable holds.
"""
import json

import pandas as pd

from src.serving import siem_platform_connector as sc

CFG = {
    "routing": {"noise_destination": "data_lake", "alert_destination": "priority_queue",
                "escalate_above": 0.90, "never_autoclose_high_confidence": True,
                "audit_sample_rate": 0.5},
    "decile_tiering": {"n_deciles": 10, "default_action": "deprioritize",
                       "actions": {"10": "escalate", "9": "priority_review",
                                   "8": "review"}},
    "siem": {"risk_index": "risk", "score_field": "triage_priority"},
}


def test_tier_comes_from_config():
    assert sc.tier_for_decile(10, CFG) == "escalate"
    assert sc.tier_for_decile(9, CFG) == "priority_review"
    assert sc.tier_for_decile(3, CFG) == "deprioritize"


def test_destination_follows_the_verdict():
    assert sc.destination("ALERT", CFG) == "priority_queue"
    assert sc.destination("NORMAL", CFG) == "data_lake"


def test_alert_record_carries_a_queue_position():
    r = sc.to_risk_record(0, 0.95, "ALERT", 88.5, 10, CFG)
    assert r["triage_priority"] == 88.5 and r["decile"] == 10
    assert r["action"] == "escalate" and r["destination"] == "priority_queue"
    assert r["escalate"] is True


def test_normal_record_has_no_queue_position():
    r = sc.to_risk_record(1, 0.01, "NORMAL", None, None, CFG)
    assert r["triage_priority"] is None and r["decile"] is None
    assert r["destination"] == "data_lake"
    assert r["escalate"] is False


def test_nan_rank_becomes_none_not_a_crash():
    """A pandas float column cannot hold None, so unranked rows arrive as NaN."""
    r = sc.to_risk_record(2, 0.02, "NORMAL", float("nan"), float("nan"), CFG)
    assert r["triage_priority"] is None and r["decile"] is None


def test_records_are_valid_json():
    """NaN would serialise as a bare NaN token, which no JSON parser accepts."""
    r = sc.to_risk_record(3, 0.02, "NORMAL", float("nan"), float("nan"), CFG)
    assert json.loads(json.dumps(r))["decile"] is None


def test_high_confidence_flow_is_never_auto_closable():
    """HITL (plan §6/§10): confidence above the escalation bar blocks auto-close."""
    r = sc.to_risk_record(4, 0.99, "NORMAL", None, None, CFG)
    assert r["auto_closable"] is False


def test_low_confidence_normal_flow_is_auto_closable():
    r = sc.to_risk_record(5, 0.01, "NORMAL", None, None, CFG)
    assert r["auto_closable"] is True


def test_an_alert_is_never_auto_closable():
    r = sc.to_risk_record(6, 0.80, "ALERT", 50.0, 5, CFG)
    assert r["auto_closable"] is False


def _scored():
    return pd.DataFrame({
        "probability": [0.99, 0.80, 0.02, 0.01],
        "verdict": ["ALERT", "ALERT", "NORMAL", "NORMAL"],
        "priority": [95.0, 60.0, float("nan"), float("nan")],
        "decile": [10.0, 6.0, float("nan"), float("nan")],
    })


def test_to_risk_records_covers_every_row():
    recs = sc.to_risk_records(_scored(), CFG, seed=0)
    assert len(recs) == 4
    assert [r["verdict"] for r in recs] == ["ALERT", "ALERT", "NORMAL", "NORMAL"]


def test_audit_sampling_marks_only_suppressed_traffic():
    """Suppression has to stay measurable, so a share of NORMAL is retained."""
    recs = sc.to_risk_records(_scored(), CFG, seed=0)
    sampled = [r for r in recs if r["audit_sampled"]]
    assert sampled, "audit_sample_rate 0.5 selected nothing"
    assert all(r["verdict"] == "NORMAL" for r in sampled)


def test_write_risk_index_is_one_json_object_per_line(tmp_path):
    out = tmp_path / "risk.jsonl"
    recs = sc.to_risk_records(_scored(), CFG, seed=0)
    sc.write_risk_index(recs, str(out))
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(recs)
    assert all(json.loads(ln)["risk_index"] == "risk" for ln in lines)


def test_summarise_counts_the_routing_split():
    s = sc.summarise(sc.to_risk_records(_scored(), CFG, seed=0))
    assert s["total"] == 4 and s["alerts"] == 2 and s["normal"] == 2
    assert s["escalate"] == 1          # only p=0.99 clears escalate_above 0.90
