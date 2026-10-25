import time

import pytest


def test_health_reports_model_and_metrics(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("1.0.0")
    assert body["metrics"]["roc_auc"] > 0.7


def test_predict_returns_prediction_and_confidence(client, engaged_user):
    r = client.post("/predict", json=engaged_user)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("likely_to_purchase", "unlikely_to_purchase")
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["latency_ms"] >= 0


def test_engaged_user_scores_higher_than_inactive_user(client, engaged_user, inactive_user):
    hot = client.post("/predict", json=engaged_user).json()
    cold = client.post("/predict", json=inactive_user).json()
    assert hot["prediction"] == "likely_to_purchase"
    assert cold["prediction"] == "unlikely_to_purchase"
    assert hot["purchase_probability"] > cold["purchase_probability"]


def test_same_input_gives_same_prediction(client, engaged_user):
    first = client.post("/predict", json=engaged_user).json()
    for _ in range(5):
        again = client.post("/predict", json=engaged_user).json()
        assert again["purchase_probability"] == first["purchase_probability"]


@pytest.mark.parametrize("field,value", [
    ("sessions_last_7d", -1),
    ("avg_session_minutes", 10_000),
    ("platform", "windows"),
    ("push_enabled", "maybe"),
])
def test_invalid_input_is_rejected(client, engaged_user, field, value):
    r = client.post("/predict", json={**engaged_user, field: value})
    assert r.status_code == 422


def test_missing_field_is_rejected(client, engaged_user):
    body = dict(engaged_user)
    del body["platform"]
    assert client.post("/predict", json=body).status_code == 422


def test_latency_well_under_two_seconds(client, engaged_user):
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        assert client.post("/predict", json=engaged_user).status_code == 200
        timings.append(time.perf_counter() - start)
    timings.sort()
    p95 = timings[int(len(timings) * 0.95) - 1]
    assert p95 < 2.0  # acceptance criterion; in practice a few milliseconds
