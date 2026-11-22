"""Modèles de prévision de la consommation électrique journalière.

Tous les modèles partagent la même interface :

    modele.fit(serie)                    # apprend sur une série quotidienne
    modele.predict(historique, horizon)  # prévoit `horizon` jours après l'historique

Cette interface unique sert au backtest, à la sélection des modèles et à l'API.
Une série est une `pandas.Series` de float indexée par des dates quotidiennes consécutives.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd

PAYS_JOURS_FERIES = "DE"


# --------------------------------------------------------------------------- utilitaires


def check_series(series: pd.Series, minimum: int = 1) -> None:
    """Vérifie qu'une série est quotidienne, continue, sans valeur manquante."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("La série doit être indexée par des dates.")
    if len(series) < minimum:
        raise ValueError(f"Historique trop court : {len(series)} jours, minimum {minimum}.")
    if series.isna().any():
        raise ValueError("La série contient des valeurs manquantes.")
    if len(series) > 1 and (series.index.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("La série doit contenir des jours consécutifs, sans trou.")


def future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Dates des `horizon` jours qui suivent la fin de l'historique."""
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un jour.")
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


@lru_cache(maxsize=64)
def _jours_feries(annee: int) -> frozenset:
    import holidays

    return frozenset(holidays.country_holidays(PAYS_JOURS_FERIES, years=annee).keys())


def holiday_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 pour un jour férié national allemand, 0.0 sinon."""
    return np.array([1.0 if jour.date() in _jours_feries(jour.year) else 0.0 for jour in index])


def calendar_features(index: pd.DatetimeIndex, fourier_order: int = 2) -> pd.DataFrame:
    """Variables calendaires : saisonnalité annuelle (Fourier), jour de semaine, jour férié."""
    angle = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
    colonnes: dict[str, np.ndarray] = {}
    for k in range(1, fourier_order + 1):
        colonnes[f"year_sin_{k}"] = np.sin(k * angle)
        colonnes[f"year_cos_{k}"] = np.cos(k * angle)
    jour = index.dayofweek.to_numpy()
    for j in range(1, 7):  # lundi = référence
        colonnes[f"dow_{j}"] = (jour == j).astype(float)
    colonnes["holiday"] = holiday_flags(index)
    return pd.DataFrame(colonnes, index=index)


# --------------------------------------------------------------------------- interface


class Forecaster(ABC):
    """Interface commune à tous les modèles."""

    name: str = "abstract"
    min_history: int = 1

    @abstractmethod
    def fit(self, series: pd.Series) -> Forecaster:
        """Apprend sur une série quotidienne et retourne le modèle."""

    @abstractmethod
    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Prévision brute, l'historique ayant déjà été validé."""

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Prévoit `horizon` jours après la dernière date de `history`."""
        check_series(history, self.min_history)
        if horizon < 1:
            raise ValueError("L'horizon doit être d'au moins un jour.")
        valeurs = np.asarray(self._predict(history, horizon), dtype=float)
        if valeurs.shape != (horizon,) or not np.isfinite(valeurs).all():
            raise RuntimeError(f"Prévision invalide produite par le modèle {self.name}.")
        return valeurs

    def get_params(self) -> dict:
        """Hyperparamètres journalisés dans MLflow."""
        return {
            cle: valeur
            for cle, valeur in vars(self).items()
            if not cle.endswith("_") and not cle.startswith("_")
        }


# --------------------------------------------------------------------------- modèles


class SeasonalNaiveForecaster(Forecaster):
    """Référence : chaque jour reprend la valeur du même jour de la semaine précédente."""

    name = "naive"

    def __init__(self, season: int = 7):
        self.season = int(season)
        self.min_history = self.season

    def fit(self, series: pd.Series) -> SeasonalNaiveForecaster:
        check_series(series, self.season)
        return self

    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        derniere_saison = history.to_numpy()[-self.season :]
        repetitions = math.ceil(horizon / self.season)
        return np.tile(derniere_saison, repetitions)[:horizon]


class RidgeForecaster(Forecaster):
    """Régression Ridge sur variables calendaires et valeurs retardées, prévue de façon récursive."""

    name = "ridge"

    def __init__(self, lags=(1, 7, 14), alpha: float = 1.0, fourier_order: int = 3):
        self.lags = [int(lag) for lag in lags]
        self.alpha = float(alpha)
        self.fourier_order = int(fourier_order)
        self.min_history = max(self.lags)

    def _design(self, series: pd.Series) -> pd.DataFrame:
        frame = calendar_features(series.index, self.fourier_order)
        for lag in self.lags:
            frame[f"lag_{lag}"] = series.shift(lag)
        return frame

    def fit(self, series: pd.Series) -> RidgeForecaster:
        from sklearn.linear_model import Ridge

        check_series(series, max(self.lags) + 30)
        design = self._design(series).dropna()
        self.feature_columns_ = list(design.columns)
        self.model_ = Ridge(alpha=self.alpha).fit(design.to_numpy(), series.loc[design.index])
        return self

    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        dates = future_index(history, horizon)
        calendrier = calendar_features(dates, self.fourier_order).to_numpy()
        coefficients = self.model_.coef_
        intercept = float(self.model_.intercept_)
        valeurs = list(history.to_numpy()[-max(self.lags) :])
        prevision = []
        for pas in range(horizon):
            retards = [valeurs[-lag] for lag in self.lags]
            ligne = np.concatenate([calendrier[pas], retards])
            valeur = intercept + float(coefficients @ ligne)
            prevision.append(valeur)
            valeurs.append(valeur)
        return np.array(prevision)


class ArimaForecaster(Forecaster):
    """ARIMA avec régresseurs calendaires (régression harmonique dynamique, statsmodels)."""

    name = "arima"

    def __init__(self, order=(2, 0, 1), fourier_order: int = 3):
        self.order = tuple(int(valeur) for valeur in order)
        self.fourier_order = int(fourier_order)
        self.min_history = 30

    def fit(self, series: pd.Series) -> ArimaForecaster:
        from statsmodels.tsa.arima.model import ARIMA

        check_series(series, 60)
        exog = calendar_features(series.index, self.fourier_order).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(series.to_numpy(), exog=exog, order=self.order, trend="c").fit()
        return self

    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        exog = calendar_features(history.index, self.fourier_order).to_numpy()
        futur = calendar_features(future_index(history, horizon), self.fourier_order).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Mêmes coefficients, filtre de Kalman réappliqué sur l'historique fourni.
            etat = self.result_.apply(history.to_numpy(), exog=exog, refit=False)
            return np.asarray(etat.forecast(steps=horizon, exog=futur))


class ProphetForecaster(Forecaster):
    """Prophet : tendance par morceaux, saisonnalités hebdomadaire et annuelle, jours fériés."""

    name = "prophet"

    def __init__(self, changepoint_prior_scale: float = 0.05, yearly_fourier_order: int = 10):
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.yearly_fourier_order = int(yearly_fourier_order)
        self.min_history = 1

    def fit(self, series: pd.Series) -> ProphetForecaster:
        from prophet import Prophet

        for journal in ("prophet", "cmdstanpy"):
            logging.getLogger(journal).setLevel(logging.WARNING)
        check_series(series, 60)
        modele = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            yearly_seasonality=self.yearly_fourier_order,
            weekly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        modele.add_country_holidays(country_name=PAYS_JOURS_FERIES)
        modele.fit(pd.DataFrame({"ds": series.index, "y": series.to_numpy()}))
        self.model_ = modele
        return self

    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        # Prophet ne dépend que des dates : l'historique fixe seulement le point de départ.
        dates = future_index(history, horizon)
        return self.model_.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy()

    def __getstate__(self) -> dict:
        # Sérialisation JSON officielle de Prophet, plus robuste que le pickle brut.
        from prophet.serialize import model_to_json

        etat = dict(self.__dict__)
        if "model_" in etat:
            etat["model_"] = model_to_json(etat["model_"])
        return etat

    def __setstate__(self, etat: dict) -> None:
        if isinstance(etat.get("model_"), str):
            from prophet.serialize import model_from_json

            etat["model_"] = model_from_json(etat["model_"])
        self.__dict__.update(etat)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_forward(weights: dict, window: np.ndarray) -> float:
    """Passe avant d'un LSTM à une couche + couche linéaire, en NumPy pur.

    Reproduit exactement `torch.nn.LSTM` (portes i, f, g, o) : l'API n'a donc pas
    besoin de PyTorch, ce qui garde son image Docker légère.
    """
    w_ih, w_hh = weights["w_ih"], weights["w_hh"]
    biais = weights["b_ih"] + weights["b_hh"]
    taille = w_hh.shape[1]
    h = np.zeros(taille)
    c = np.zeros(taille)
    for x_t in window:
        portes = w_ih @ x_t + w_hh @ h + biais
        i = _sigmoid(portes[:taille])
        f = _sigmoid(portes[taille : 2 * taille])
        g = np.tanh(portes[2 * taille : 3 * taille])
        o = _sigmoid(portes[3 * taille :])
        c = f * c + i * g
        h = o * np.tanh(c)
    return float(weights["w_out"] @ h + weights["b_out"])


class LstmForecaster(Forecaster):
    """LSTM entraîné avec PyTorch ; prévision récursive jour par jour en NumPy."""

    name = "lstm"

    def __init__(
        self,
        window: int = 28,
        hidden_size: int = 32,
        epochs: int = 40,
        learning_rate: float = 0.005,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.window = int(window)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_history = self.window

    def _inputs(self, values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        """Entrées par jour : valeur normalisée, jour de semaine, saison, jour férié."""
        z = (values - self.mean_) / self.std_
        semaine = 2 * np.pi * dates.dayofweek.to_numpy() / 7
        annee = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
        return np.column_stack(
            [z, np.sin(semaine), np.cos(semaine), np.sin(annee), np.cos(annee), holiday_flags(dates)]
        )

    def fit(self, series: pd.Series) -> LstmForecaster:
        import torch
        from torch import nn

        check_series(series, self.window + 60)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # calcul déterministe d'une exécution à l'autre

        self.mean_ = float(series.mean())
        self.std_ = float(series.std())
        entrees = self._inputs(series.to_numpy(), series.index)
        cibles = (series.to_numpy() - self.mean_) / self.std_
        x = np.stack([entrees[i : i + self.window] for i in range(len(series) - self.window)])
        y = cibles[self.window :]

        lstm = nn.LSTM(entrees.shape[1], self.hidden_size, batch_first=True)
        sortie = nn.Linear(self.hidden_size, 1)
        parametres = list(lstm.parameters()) + list(sortie.parameters())
        optimiseur = torch.optim.Adam(parametres, lr=self.learning_rate)
        perte_fn = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        generateur = torch.Generator().manual_seed(self.seed)

        try:
            for _ in range(self.epochs):
                ordre = torch.randperm(len(x_t), generator=generateur)
                for debut in range(0, len(ordre), self.batch_size):
                    lot = ordre[debut : debut + self.batch_size]
                    optimiseur.zero_grad()
                    etats, _ = lstm(x_t[lot])
                    perte = perte_fn(sortie(etats[:, -1, :]), y_t[lot])
                    perte.backward()
                    optimiseur.step()
            with torch.no_grad():
                etats, _ = lstm(x_t)
                self.train_loss_ = float(perte_fn(sortie(etats[:, -1, :]), y_t))
        finally:
            torch.set_num_threads(threads)

        self.weights_ = {
            "w_ih": lstm.weight_ih_l0.detach().numpy().astype(float),
            "w_hh": lstm.weight_hh_l0.detach().numpy().astype(float),
            "b_ih": lstm.bias_ih_l0.detach().numpy().astype(float),
            "b_hh": lstm.bias_hh_l0.detach().numpy().astype(float),
            "w_out": sortie.weight.detach().numpy().astype(float)[0],
            "b_out": float(sortie.bias.detach().numpy()[0]),
        }
        return self

    def _predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        valeurs = list(history.to_numpy()[-self.window :])
        dates = list(history.index[-self.window :])
        prevision = []
        for date in future_index(history, horizon):
            entrees = self._inputs(np.array(valeurs[-self.window :]), pd.DatetimeIndex(dates[-self.window :]))
            valeur = lstm_forward(self.weights_, entrees) * self.std_ + self.mean_
            prevision.append(valeur)
            valeurs.append(valeur)
            dates.append(date)
        return np.array(prevision)


# --------------------------------------------------------------------------- fabrique

MODELES: dict[str, type[Forecaster]] = {
    classe.name: classe
    for classe in (
        SeasonalNaiveForecaster,
        RidgeForecaster,
        ArimaForecaster,
        ProphetForecaster,
        LstmForecaster,
    )
}


def build_forecaster(name: str, params: dict | None = None) -> Forecaster:
    """Instancie un modèle à partir de son nom et de sa section de params.yaml."""
    if name not in MODELES:
        raise ValueError(f"Modèle inconnu : {name}. Disponibles : {sorted(MODELES)}.")
    return MODELES[name](**(params or {}))
