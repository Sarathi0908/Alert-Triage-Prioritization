"""Tests for src/serving/flow_triage_api.py.

The promise of this API is that the request body IS the feature vector — no hidden
derivation — and that the served operating point is the evaluated one, because the
threshold and the priority grids travel inside the model bundle. Both are tested.
"""


def test_health_reports_the_bundle_contract(client, trained):
    model, _ = trained
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] == len(model.feature_names)
    assert body["threshold_alert"] == round(model.threshold, 6)


def test_served_threshold_matches_the_trained_run(client, metrics):
    """The served operating point must not drift from the one that was evaluated."""
    assert client.get("/health").json()["threshold_alert"] == metrics["threshold"]


def test_feature_contract_declares_no_derived_features(client, trained):
    model, _ = trained
    body = client.get("/features").json()
    assert body["request_body_accepts_NATIVE_only"] == model.feature_names
    assert body["derived"] == []


def test_example_body_is_a_complete_feature_vector(client, trained):
    model, _ = trained
    assert set(client.get("/example").json()) == set(model.feature_names)


def test_example_body_scores(client):
    r = client.post("/score", json=client.get("/example").json())
    assert r.status_code == 200
    assert r.json()["verdict"] in ("ALERT", "NORMAL")


def test_metrics_endpoint_serves_the_training_metrics(client, metrics):
    assert client.get("/metrics").json()["roc_auc"] == metrics["roc_auc"]


def test_score_response_shape(client, benign_flow):
    body = client.post("/score", json=benign_flow).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["verdict"] in ("ALERT", "NORMAL")
    assert body["top_factors"], "no SHAP drivers returned"
    for f in body["top_factors"]:
        assert f["pushes"] in ("ALERT", "NORMAL")


def test_verdict_agrees_with_the_threshold(client, benign_flow, trained):
    model, _ = trained
    body = client.post("/score", json=benign_flow).json()
    expected = "ALERT" if body["probability"] >= model.threshold else "NORMAL"
    assert body["verdict"] == expected


def test_normal_flows_are_not_ranked(client, benign_flow, cfg):
    """Priority and decile exist for alerts only — normal traffic is not queued."""
    body = client.post("/score", json=benign_flow).json()
    if body["verdict"] == "NORMAL":
        assert body["priority"] is None and body["decile"] is None
        assert body["destination"] == cfg["serving"]["routing"]["noise_destination"]
    else:
        assert 0.0 <= body["priority"] <= 100.0
        assert 1 <= body["decile"] <= cfg["serving"]["decile_tiering"]["n_deciles"]
        assert body["destination"] == cfg["serving"]["routing"]["alert_destination"]


def test_missing_features_default_rather_than_error(client):
    """A partial body must still score — missing fields fall back to the default."""
    r = client.post("/score", json={})
    assert r.status_code == 200
    assert r.json()["verdict"] in ("ALERT", "NORMAL")


def test_unknown_fields_are_ignored(client, benign_flow):
    """Extra keys must not shift the score — the feature list is the contract."""
    base = client.post("/score", json=benign_flow).json()["probability"]
    noisy = client.post("/score", json={**benign_flow, "NOT_A_FEATURE": 12345}
                        ).json()["probability"]
    assert base == noisy


def test_a_high_confidence_flow_is_escalated(client, trained):
    """A real attack-shaped flow should clear the escalation bar and be queued."""
    model, blob = trained
    loud = {f: 0 for f in model.feature_names}
    for f in model.feature_names:
        if "PORT" in f.upper():
            loud[f] = 80
        elif "THROUGHPUT" in f.upper() or "SECOND_BYTES" in f.upper():
            loud[f] = 10_000_000
        elif "DURATION" in f.upper():
            loud[f] = 4_000_000
        elif "TTL" in f.upper():
            loud[f] = 63
    body = client.post("/score", json=loud).json()
    assert body["verdict"] in ("ALERT", "NORMAL")
    if body["verdict"] == "ALERT":
        assert body["action"] is not None
        assert body["destination"] != "data_lake"


def test_batch_scores_every_flow_and_ranks_alerts_first(client, benign_flow):
    r = client.post("/score/batch", json=[benign_flow, benign_flow, benign_flow])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and len(body["ranked"]) == 3
    verdicts = [x["verdict"] for x in body["ranked"]]
    assert verdicts == sorted(verdicts, key=lambda v: v != "ALERT"), (
        "alerts must be ranked ahead of normal flows")


def test_batch_rejects_an_empty_body(client):
    assert client.post("/score/batch", json=[]).status_code == 422


def test_batch_agrees_with_single_scoring(client, benign_flow):
    single = client.post("/score", json=benign_flow).json()["probability"]
    batch = client.post("/score/batch", json=[benign_flow]).json()["ranked"][0]["probability"]
    assert single == batch
