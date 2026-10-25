"""Loads the model once at startup and keeps it in memory, so each request is a single fast inference."""
import os
from pathlib import Path

import joblib
import numpy as np

from app.schemas import PredictRequest
from app.training import train_and_save

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))


class Predictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.version = bundle["version"]
        self.metrics = bundle["metrics"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Predictor":
        """Load the saved model; train the demo model first if none exists yet."""
        path = Path(path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH)
        bundle = joblib.load(path) if path.exists() else train_and_save(path)
        return cls(bundle)

    @staticmethod
    def to_row(req: PredictRequest) -> np.ndarray:
        return np.array([[
            req.sessions_last_7d,
            req.avg_session_minutes,
            req.days_since_signup,
            req.purchases_last_30d,
            1.0 if req.push_enabled else 0.0,
            1.0 if req.platform == "ios" else 0.0,
        ]])

    def predict(self, req: PredictRequest) -> tuple[str, float, float]:
        """(label, confidence in that label, raw purchase probability)."""
        p = float(self.model.predict_proba(self.to_row(req))[0, 1])
        if p >= THRESHOLD:
            return "likely_to_purchase", p, p
        return "unlikely_to_purchase", 1 - p, p
