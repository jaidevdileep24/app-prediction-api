import csv

import pytest

import train as train_cli
from app.predictor import Predictor
from app.schemas import PredictRequest
from app.training import FEATURES, LABEL, load_csv, synthetic_data, train


def test_model_beats_chance_on_held_out_data():
    X, y = synthetic_data(n=2_000)
    metrics = train(X, y)["metrics"]
    assert metrics["roc_auc"] > 0.75
    assert metrics["accuracy"] > 0.65


def test_predictor_trains_a_model_when_none_exists(tmp_path):
    path = tmp_path / "missing" / "model.joblib"
    p = Predictor.load(path)
    assert path.exists()
    assert p.features == FEATURES


def test_saved_model_loads_back_identically(tmp_path):
    path = tmp_path / "model.joblib"
    first = Predictor.load(path)
    second = Predictor.load(path)
    req = PredictRequest(sessions_last_7d=5, avg_session_minutes=6, days_since_signup=90,
                         purchases_last_30d=1, push_enabled=True, platform="android")
    assert first.predict(req) == second.predict(req)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_training_from_csv(tmp_path):
    X, y = synthetic_data(n=400)
    rows = []
    for x, label in zip(X, y):
        row = dict(zip(FEATURES, x))
        row["platform"] = "ios" if row.pop("platform_ios") else "android"
        row["push_enabled"] = "true" if row["push_enabled"] else "false"
        row[LABEL] = int(label)
        rows.append(row)
    data = tmp_path / "data.csv"
    _write_csv(data, rows)

    X2, y2 = load_csv(data)
    assert X2.shape == (400, len(FEATURES))
    assert (X2[:, FEATURES.index("platform_ios")] == X[:, FEATURES.index("platform_ios")]).all()

    out = tmp_path / "csv_model.joblib"
    bundle = train_cli.main(["--csv", str(data), "--out", str(out)])
    assert out.exists()
    assert bundle["metrics"]["test_rows"] == 80


def test_empty_csv_is_an_error(tmp_path):
    data = tmp_path / "empty.csv"
    data.write_text(",".join(FEATURES + [LABEL]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv(data)
