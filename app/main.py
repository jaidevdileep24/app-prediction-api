"""FastAPI service: one POST /predict endpoint for the mobile app, plus /health for load balancers."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.predictor import Predictor
from app.schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor.load()
    yield


app = FastAPI(
    title="App Prediction API",
    version="1.0.0",
    description="Real-time purchase prediction for the mobile app: send a user's recent activity, "
                "get back a prediction and a confidence score.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    p: Predictor = request.app.state.predictor
    return HealthResponse(status="ok", model_version=p.version, metrics=p.metrics)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    start = time.perf_counter()
    p: Predictor = request.app.state.predictor
    label, confidence, probability = p.predict(body)
    return PredictResponse(
        prediction=label,
        confidence=round(confidence, 4),
        purchase_probability=round(probability, 4),
        model_version=p.version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
