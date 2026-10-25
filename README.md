# App Prediction API

A self-contained prediction service for a mobile app: the app sends one user's recent in-app activity to a
single endpoint and gets back a **prediction with a confidence score** in a few milliseconds.

This demo predicts whether a user is likely to make a purchase in the coming days. The same structure works
for any yes/no or category prediction (churn, conversion, next action...). You swap in your own anonymised
data and retrain with a single command.

```
React Native app ──POST /predict──▶ FastAPI ──▶ scikit-learn model (loaded once, kept in memory)
                 ◀── prediction + confidence ──┘
```

## Run it

```bash
docker compose up --build
```

Then open **http://localhost:8000/docs** for interactive API docs, or call it directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sessions_last_7d": 12, "avg_session_minutes": 9.5, "days_since_signup": 45,
       "purchases_last_30d": 2, "push_enabled": true, "platform": "android"}'
```

```json
{
  "prediction": "likely_to_purchase",
  "confidence": 0.9669,
  "purchase_probability": 0.9669,
  "model_version": "1.0.0-20261026",
  "latency_ms": 2.42
}
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/predict` | Prediction + confidence for one user. Invalid input gets a clear `422` error |
| `GET` | `/health` | Model version and its held-out accuracy / ROC AUC (for load balancers and monitoring) |
| `GET` | `/docs` | Swagger UI generated from the request/response schemas |

## Performance

Measured locally, 500 sequential HTTP calls to `/predict` (FastAPI + uvicorn):

| p50 | p95 | p99 |
|---|---|---|
| 7.5 ms | 9.9 ms | 12.7 ms |

The model is loaded once at startup, so each request is a single in-memory inference. That leaves plenty of
headroom under a 2-second budget, even on a small AWS t3.medium.

## Model

- Gradient boosting classifier (scikit-learn), trained at Docker build time
- 80/20 train/test split. On the demo data: **accuracy 0.785, ROC AUC 0.836** on held-out users
- Model, feature list, version and metrics are saved together in one file, so `/health` always reports the
  model actually serving

### Train on your own data

Provide a CSV with one row per user, the feature columns and a `purchased` 0/1 label:

```
sessions_last_7d,avg_session_minutes,days_since_signup,purchases_last_30d,push_enabled,platform,purchased
12,9.5,45,2,true,android,1
```

```bash
python train.py --csv your_data.csv
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

15 tests covering valid predictions, consistency, input validation, latency, training and CSV loading.
**Coverage: 99%** (the build fails below 90%). GitHub Actions runs the tests, builds the Docker image and
calls the live container on every push.

A Postman collection with its own tests is in `postman/`: import it, run `docker compose up`, then Run Collection.

## Deploy to AWS

On an EC2 instance (t3.medium is plenty) with Docker installed:

```bash
git clone <this repo> && cd app-prediction-api
docker compose up -d --build
```

Open port 8000 in the security group, or put it behind an Application Load Balancer that uses `/health`
as its health check.

## Project structure

```
app/
  main.py        FastAPI app: /predict and /health
  schemas.py     request/response contracts and input validation
  predictor.py   loads the model once and runs inference
  training.py    data loading, training and evaluation
train.py         CLI: python train.py [--csv data.csv]
tests/           pytest suite (API + training)
postman/         Postman collection with tests
Dockerfile, docker-compose.yml
```
