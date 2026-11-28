"""Validation du modèle réellement produit par le pipeline (après `dvc repro`).

Ces tests sont ignorés tant que l'artefact n'existe pas ; en CI ils s'exécutent
après le pipeline et bloquent une version dont les sorties seraient invalides.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common import ROOT, load_params

MODEL_PATH = ROOT / "models" / "forecaster.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not METRICS_PATH.exists(),
    reason="Artefact absent : lancer `dvc repro` d'abord.",
)


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def params():
    return load_params()


def test_selected_models_meet_the_performance_thresholds(params):
    rapport = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert rapport["status"] == "passed", rapport["failures"]
    for terme in ("short", "long"):
        seuil = params["evaluation"][f"max_mape_{terme}"]
        assert rapport["selected"][terme]["mape"] <= seuil


def test_selected_models_beat_the_naive_baseline():
    rapport = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for terme in ("short", "long"):
        choisi = rapport["selected"][terme]
        assert choisi["mae"] <= rapport["models"]["naive"][terme]["mae"]


@pytest.mark.parametrize("horizon", [1, 7, 14, 15, 90, 180])
def test_forecast_format_and_plausible_values(bundle, params, horizon):
    terme = "short_term" if horizon <= bundle["short_term_max_days"] else "long_term"
    prevision = bundle[terme]["forecaster"].predict(bundle["history"], horizon)
    assert prevision.shape == (horizon,)
    assert np.isfinite(prevision).all()
    assert (prevision > params["data"]["min_gwh"]).all()
    assert (prevision < params["data"]["max_gwh"]).all()


def test_forecast_is_deterministic(bundle):
    modele = bundle["long_term"]["forecaster"]
    np.testing.assert_allclose(modele.predict(bundle["history"], 30), modele.predict(bundle["history"], 30))


def test_bundle_is_traceable_and_starts_after_last_observation(bundle):
    assert bundle["trace"]["mlflow_run_id"]
    assert len(bundle["trace"]["data_train_md5"]) == 32
    historique = bundle["history"]
    assert historique.index[-1] == pd.Timestamp("2017-12-31")
    assert not historique.isna().any()
