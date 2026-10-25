"""Train the purchase-prediction model and save it as one joblib bundle.

Data: either a CSV of anonymised in-app data (one row per user, the FEATURES columns plus a 0/1
`purchased` label) or, for the demo, a synthetic dataset with the same shape.
"""
import csv
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "sessions_last_7d",
    "avg_session_minutes",
    "days_since_signup",
    "purchases_last_30d",
    "push_enabled",
    "platform_ios",
]
LABEL = "purchased"
MODEL_VERSION = f"1.0.0-{date.today():%Y%m%d}"


def synthetic_data(n: int = 5_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Realistic-looking user activity where engagement and past purchases drive the outcome."""
    rng = np.random.default_rng(seed)
    sessions = rng.poisson(6, n)
    minutes = rng.gamma(2.0, 4.0, n)
    days = rng.integers(0, 720, n)
    purchases = rng.poisson(0.8, n)
    push = rng.binomial(1, 0.6, n)
    ios = rng.binomial(1, 0.35, n)
    logit = (-4.6 + 0.35 * sessions + 0.12 * minutes + 1.1 * purchases
             - 0.003 * days + 1.0 * push + 0.3 * ios + rng.normal(0, 0.3, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    X = np.column_stack([sessions, minutes, days, purchases, push, ios]).astype(float)
    return X, y


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read FEATURES + LABEL columns; `platform` (android/ios) is accepted instead of platform_ios."""
    rows, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "platform_ios" not in r:
                r["platform_ios"] = 1 if r.get("platform", "").strip().lower() == "ios" else 0
            if str(r["push_enabled"]).strip().lower() in ("true", "false"):
                r["push_enabled"] = 1 if str(r["push_enabled"]).strip().lower() == "true" else 0
            rows.append([float(r[c]) for c in FEATURES])
            labels.append(int(float(r[LABEL])))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return np.array(rows), np.array(labels)


def train(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Fit on 80%, report accuracy/ROC AUC on the held-out 20%, then return the bundle to save."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, proba >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "train_rows": float(len(y_tr)),
        "test_rows": float(len(y_te)),
    }
    return {"model": model, "features": FEATURES, "version": MODEL_VERSION, "metrics": metrics}


def train_and_save(path: str | Path, csv_path: str | Path | None = None) -> dict:
    X, y = load_csv(csv_path) if csv_path else synthetic_data()
    bundle = train(X, y)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle
