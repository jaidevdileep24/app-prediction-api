"""Request/response contracts for the prediction endpoint.

The mobile app sends one user's recent in-app activity; the API answers with a prediction and a
confidence score. Field limits reject obviously broken input with a 422 before it reaches the model.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sessions_last_7d: int = Field(..., ge=0, le=500, description="App sessions in the last 7 days")
    avg_session_minutes: float = Field(..., ge=0, le=600, description="Average session length in minutes")
    days_since_signup: int = Field(..., ge=0, le=10_000, description="Days since the account was created")
    purchases_last_30d: int = Field(..., ge=0, le=1_000, description="Purchases in the last 30 days")
    push_enabled: bool = Field(..., description="Push notifications switched on")
    platform: Literal["android", "ios"]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "sessions_last_7d": 12,
                "avg_session_minutes": 9.5,
                "days_since_signup": 45,
                "purchases_last_30d": 2,
                "push_enabled": True,
                "platform": "android",
            }]
        }
    }


class PredictResponse(BaseModel):
    prediction: Literal["likely_to_purchase", "unlikely_to_purchase"]
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted outcome")
    purchase_probability: float = Field(..., ge=0, le=1)
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    metrics: dict[str, float]
