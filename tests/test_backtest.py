"""Backtest à origine glissante."""

import numpy as np

from src.backtest import rolling_origin
from src.forecasters import Forecaster


class Oracle(Forecaster):
    """Connaît la série complète : son erreur doit être nulle."""

    name = "oracle"

    def __init__(self, truth):
        self.truth = truth

    def fit(self, series):
        return self

    def _predict(self, history, horizon):
        debut = self.truth.index.get_loc(history.index[-1]) + 1
        return self.truth.to_numpy()[debut : debut + horizon]


def test_rolling_origin_uses_only_the_past_and_counts_origins(series):
    train, test = series.iloc[:-60], series.iloc[-60:]
    metriques, detail = rolling_origin(Oracle(series), train, test, horizon=7, step=7)
    assert metriques["mae"] == 0.0
    assert metriques["n_origins"] == 8  # décalages 0, 7, ..., 49 ; le suivant dépasserait
    assert (detail["date"] > detail["origin"]).all()
    assert np.isclose(detail["actual"], detail["forecast"]).all()


def test_horizon_longer_than_test_is_capped(series):
    train, test = series.iloc[:-30], series.iloc[-30:]
    metriques, detail = rolling_origin(Oracle(series), train, test, horizon=180, step=30)
    assert metriques["n_origins"] == 1
    assert len(detail) == 30
