"""Tests de l'API dans les trois environnements, avec un petit modèle de test."""

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import create_app, load_environment_config
from src.forecasters import build_forecaster
from tests.conftest import make_series


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    historique = make_series()
    naif = build_forecaster("naive").fit(historique)
    ridge = build_forecaster("ridge").fit(historique)
    bundle = {
        "model_version": "test-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "unit": "GWh",
        "short_term_max_days": 14,
        "short_term": {"name": "ridge", "forecaster": ridge},
        "long_term": {"name": "naive", "forecaster": naif},
        "backtest": {"short": {"model": "ridge", "mae": 20.0}, "long": {"model": "naive", "mae": 40.0}},
        "history": historique,
        "trace": {"mlflow_run_id": "abc", "git_commit": "inconnu"},
        "all_models": {},
    }
    chemin = tmp_path_factory.mktemp("modele") / "forecaster.joblib"
    joblib.dump(bundle, chemin)
    return chemin


def client_for(environment, model_path):
    return TestClient(create_app(load_environment_config(environment), model_path))


def test_health_ready_and_model_metadata(model_path):
    with client_for("dev", model_path) as client:
        assert client.get("/health").json() == {"status": "ok", "environment": "dev"}
        assert client.get("/ready").json()["model_version"] == "test-1"
        modele = client.get("/model").json()
        assert modele["short_term"]["model"] == "ridge"
        assert modele["long_term"]["model"] == "naive"
        assert modele["data_until"] == "2016-12-30"


def test_short_and_long_term_forecasts_use_their_own_model(model_path):
    with client_for("staging", model_path) as client:
        court = client.post("/predict", json={"horizon_days": 7}).json()
        long = client.post("/predict", json={"horizon_days": 60}).json()
    assert court["horizon_type"] == "short" and court["model"] == "ridge"
    assert long["horizon_type"] == "long" and long["model"] == "naive"
    assert len(court["forecast"]) == 7
    assert court["forecast"][0]["date"] == "2016-12-31"
    assert long["forecast"][-1]["date"] == "2017-02-28"
    assert all(point["consumption_gwh"] > 0 for point in long["forecast"])


def test_default_horizon_depends_on_environment(model_path):
    with client_for("dev", model_path) as client:
        assert len(client.post("/predict").json()["forecast"]) == 7
    with client_for("prod", model_path) as client:
        assert len(client.post("/predict", json={}).json()["forecast"]) == 30


@pytest.mark.parametrize(("environment", "maximum"), [("dev", 30), ("staging", 180), ("prod", 180)])
def test_horizon_limit_is_enforced_per_environment(model_path, environment, maximum):
    with client_for(environment, model_path) as client:
        assert client.post("/predict", json={"horizon_days": maximum}).status_code == 200
        refus = client.post("/predict", json={"horizon_days": maximum + 1})
        assert refus.status_code == 422
        assert str(maximum) in refus.json()["detail"]
        assert client.post("/predict", json={"horizon_days": 0}).status_code == 422


def test_missing_model_keeps_liveness_but_fails_readiness(tmp_path):
    with client_for("dev", tmp_path / "absent.joblib") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"horizon_days": 1}).status_code == 503


def test_unknown_environment_is_refused():
    with pytest.raises(RuntimeError, match="APP_ENV"):
        load_environment_config("recette")
