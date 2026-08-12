"""Serving-contract tests for src/serving/api.py.

The promise of this API is that the request body IS the feature vector — no hidden
derivation — and that the served operating point is the evaluated one, because the
threshold travels inside the model bundle. Both are tested here.
"""


def test_health_reports_the_bundle_contract(client, bundle):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == len(bundle["features"])
    assert body["threshold_alert"] == round(bundle["threshold"], 6)


def test_served_threshold_matches_the_trained_run(client, metrics):
    """The served operating point must not drift from the one that was evaluated."""
    assert client.get("/health").json()["threshold_alert"] == metrics["threshold"]


def test_feature_contract_declares_no_derived_features(client, bundle):
    body = client.get("/features").json()
    assert body["request_body_accepts_NATIVE_only"] == bundle["features"]
    assert body["derived"] == []


def test_example_body_is_a_complete_feature_vector(client, bundle):
    example = client.get("/example").json()
    assert set(example) == set(bundle["features"])


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


def test_verdict_agrees_with_the_threshold(client, benign_flow, bundle):
    body = client.post("/score", json=benign_flow).json()
    expected = "ALERT" if body["probability"] >= bundle["threshold"] else "NORMAL"
    assert body["verdict"] == expected


def test_normal_flows_are_not_ranked(client, benign_flow):
    """Priority and decile exist for alerts only — normal traffic is not queued."""
    body = client.post("/score", json=benign_flow).json()
    if body["verdict"] == "NORMAL":
        assert body["priority"] is None and body["decile"] is None
    else:
        assert 0.0 <= body["priority"] <= 100.0
        assert 1 <= body["decile"] <= 10


def test_missing_features_default_rather_than_error(client, cfg):
    """A partial body must still score — missing fields fall back to the configured default."""
    r = client.post("/score", json={})
    assert r.status_code == 200
    assert r.json()["verdict"] in ("ALERT", "NORMAL")


def test_unknown_fields_are_ignored(client, benign_flow):
    """Extra keys in the body must not shift the score — the feature list is the contract."""
    base = client.post("/score", json=benign_flow).json()["probability"]
    noisy = client.post("/score", json={**benign_flow, "NOT_A_FEATURE": 12345}).json()["probability"]
    assert base == noisy


def test_batch_scores_every_flow_and_ranks_alerts_first(client, benign_flow):
    r = client.post("/score/batch", json=[benign_flow, benign_flow, benign_flow])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["ranked"]) == 3
    verdicts = [x["verdict"] for x in body["ranked"]]
    assert verdicts == sorted(verdicts, key=lambda v: v != "ALERT"), (
        "alerts must be ranked ahead of normal flows")


def test_batch_rejects_an_empty_body(client):
    assert client.post("/score/batch", json=[]).status_code == 422


def test_batch_agrees_with_single_scoring(client, benign_flow):
    single = client.post("/score", json=benign_flow).json()["probability"]
    batch = client.post("/score/batch", json=[benign_flow]).json()["ranked"][0]["probability"]
    assert single == batch
