import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    # A fresh model in a temp dir, so tests never depend on (or overwrite) models/model.joblib.
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(tmp_path_factory.mktemp("model") / "model.joblib"))
    from app.main import app
    with TestClient(app) as c:  # the context manager runs the startup hook that loads the model
        yield c
    mp.undo()


@pytest.fixture
def engaged_user():
    return {"sessions_last_7d": 25, "avg_session_minutes": 15, "days_since_signup": 30,
            "purchases_last_30d": 4, "push_enabled": True, "platform": "ios"}


@pytest.fixture
def inactive_user():
    return {"sessions_last_7d": 0, "avg_session_minutes": 0.5, "days_since_signup": 600,
            "purchases_last_30d": 0, "push_enabled": False, "platform": "android"}
