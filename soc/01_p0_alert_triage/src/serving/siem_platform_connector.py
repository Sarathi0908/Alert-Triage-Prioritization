"""
siem_platform_connector.py  —  emit flow triage scores to the SIEM / SOAR layer
(plan §7: Splunk Risk-Based Alerting, SOAR queues).

The MODEL decides a score, a priority and an ALERT/NORMAL verdict. This module
decides nothing — it formats that decision into risk records the platform can
route on, and states the destination the backend should honour: NORMAL to the Data
Lake, ALERT to the analyst priority queue.

Human-in-the-loop (plan §6/§10): a high-confidence flow is never marked
auto-closable, whatever its decile, and a sample of NORMAL traffic is always
retained for audit so suppression stays measurable.
"""
from __future__ import annotations

import json
import math

import pandas as pd


def _blank_if_missing(v):
    """None for an absent rank. A pandas float column cannot hold None, so an
    unranked NORMAL flow arrives here as NaN; both mean 'no queue position', and
    NaN is not valid JSON, so it must not survive into a risk record."""
    if v is None:
        return None
    try:
        return None if math.isnan(float(v)) else v
    except (TypeError, ValueError):
        return None


def tier_for_decile(decile: int, serving_cfg: dict) -> str:
    """Analyst-queue tier from the priority decile (configs/serving_config.yaml)."""
    tiering = serving_cfg["decile_tiering"]
    return tiering["actions"].get(str(int(decile)), tiering["default_action"])


def destination(verdict: str, serving_cfg: dict) -> str:
    routing = serving_cfg["routing"]
    return (routing["alert_destination"] if verdict == "ALERT"
            else routing["noise_destination"])


def to_risk_record(index: int, probability: float, verdict: str,
                   priority: float | None, decile: int | None,
                   serving_cfg: dict) -> dict:
    """One SIEM risk record. `priority`/`decile` are None for NORMAL flows —
    normal traffic is never ranked, so it carries no queue position."""
    routing = serving_cfg["routing"]
    siem = serving_cfg["siem"]
    is_alert = verdict == "ALERT"
    escalate = is_alert and probability >= routing["escalate_above"]
    priority = _blank_if_missing(priority)
    decile = _blank_if_missing(decile)
    return {
        "flow_index": int(index),
        "risk_index": siem["risk_index"],
        siem["score_field"]: round(float(priority), 2) if priority is not None else None,
        "probability": round(float(probability), 6),
        "verdict": verdict,
        "decile": int(decile) if decile is not None else None,
        "action": (tier_for_decile(decile, serving_cfg)
                   if is_alert and decile is not None else routing["noise_destination"]),
        "destination": destination(verdict, serving_cfg),
        "escalate": bool(escalate),
        # HITL: high-confidence traffic is never a candidate for auto-close.
        "auto_closable": bool(
            (not is_alert)
            and not (routing["never_autoclose_high_confidence"]
                     and probability >= routing["escalate_above"])),
        "audit_sampled": False,
    }


def to_risk_records(scored: pd.DataFrame, serving_cfg: dict,
                    seed: int = 42) -> list[dict]:
    """Convert a scored frame (probability/verdict/priority/decile) to risk records,
    marking an audit sample of the suppressed NORMAL traffic."""
    records = [
        to_risk_record(i, r.probability, r.verdict,
                       getattr(r, "priority", None), getattr(r, "decile", None),
                       serving_cfg)
        for i, r in enumerate(scored.itertuples(index=False))
    ]
    rate = serving_cfg["routing"].get("audit_sample_rate", 0.0)
    if rate > 0:
        normal_idx = [i for i, rec in enumerate(records) if rec["verdict"] == "NORMAL"]
        if normal_idx:
            import numpy as np
            n = max(1, int(round(len(normal_idx) * rate)))
            for i in np.random.RandomState(seed).choice(normal_idx, size=min(n, len(normal_idx)),
                                                        replace=False):
                records[int(i)]["audit_sampled"] = True
    return records


def write_risk_index(records: list[dict], path: str) -> str:
    """One JSON object per line — the shape SIEM risk-index ingestion expects."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def summarise(records: list[dict]) -> dict:
    """Routing counts, for the run's metrics and for a sanity check on suppression."""
    alerts = [r for r in records if r["verdict"] == "ALERT"]
    return {"total": len(records), "alerts": len(alerts),
            "normal": len(records) - len(alerts),
            "escalate": sum(1 for r in records if r["escalate"]),
            "auto_closable": sum(1 for r in records if r["auto_closable"]),
            "audit_sampled": sum(1 for r in records if r["audit_sampled"])}
