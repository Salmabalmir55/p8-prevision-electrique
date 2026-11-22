"""Chaque modèle respecte l'interface commune et produit des prévisions valides."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.forecasters import (
    MODELES,
    LstmForecaster,
    build_forecaster,
    calendar_features,
    check_series,
    future_index,
    lstm_forward,
)

# Réglages allégés pour que les tests restent rapides.
RAPIDES = {
    "naive": {},
    "ridge": {},
    "arima": {"order": [1, 0, 0]},
    "prophet": {"yearly_fourier_order": 3},
    "lstm": {"window": 14, "hidden_size": 8, "epochs": 2},
}


@pytest.mark.parametrize("nom", sorted(MODELES))
def test_every_model_forecasts_the_requested_horizon(series, nom):
    if nom == "lstm":
        pytest.importorskip("torch")
    modele = build_forecaster(nom, RAPIDES[nom]).fit(series)
    prevision = modele.predict(series, 10)
    assert prevision.shape == (10,)
    assert np.isfinite(prevision).all()
    assert 700 < prevision.mean() < 1900  # ordre de grandeur de la série synthétique
    # L'API recharge le modèle depuis un fichier : la sérialisation ne doit rien changer.
    recharge = pickle.loads(pickle.dumps(modele))
    np.testing.assert_allclose(recharge.predict(series, 10), prevision)


def test_ridge_learns_the_weekly_pattern(series):
    modele = build_forecaster("ridge").fit(series)
    prevision = pd.Series(modele.predict(series, 14), index=future_index(series, 14))
    week_end = prevision[prevision.index.dayofweek >= 5].mean()
    semaine = prevision[prevision.index.dayofweek < 5].mean()
    assert semaine - week_end > 100


def test_lstm_numpy_inference_matches_pytorch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(6, 5, batch_first=True)
    sortie = torch.nn.Linear(5, 1)
    x = torch.randn(1, 12, 6)
    with torch.no_grad():
        attendu = float(sortie(lstm(x)[0][:, -1, :])[0, 0])
    poids = {
        "w_ih": lstm.weight_ih_l0.detach().numpy().astype(float),
        "w_hh": lstm.weight_hh_l0.detach().numpy().astype(float),
        "b_ih": lstm.bias_ih_l0.detach().numpy().astype(float),
        "b_hh": lstm.bias_hh_l0.detach().numpy().astype(float),
        "w_out": sortie.weight.detach().numpy().astype(float)[0],
        "b_out": float(sortie.bias.detach().numpy()[0]),
    }
    assert lstm_forward(poids, x[0].numpy().astype(float)) == pytest.approx(attendu, abs=1e-5)


def test_lstm_training_is_reproducible(series):
    pytest.importorskip("torch")
    a = LstmForecaster(window=14, hidden_size=8, epochs=2).fit(series).predict(series, 5)
    b = LstmForecaster(window=14, hidden_size=8, epochs=2).fit(series).predict(series, 5)
    np.testing.assert_allclose(a, b)


def test_calendar_features_flag_german_holidays():
    index = pd.date_range("2016-12-24", periods=4, freq="D")
    assert calendar_features(index)["holiday"].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_invalid_inputs_are_rejected(series):
    with pytest.raises(ValueError, match="consécutifs"):
        check_series(series.drop(series.index[5]))
    with pytest.raises(ValueError, match="manquantes"):
        check_series(series.where(series.index != series.index[3]))
    with pytest.raises(ValueError, match="horizon"):
        build_forecaster("naive").fit(series).predict(series, 0)
    with pytest.raises(ValueError, match="inconnu"):
        build_forecaster("xgboost")
