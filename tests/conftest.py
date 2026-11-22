"""Données synthétiques partagées par les tests (aucun téléchargement nécessaire)."""

import numpy as np
import pandas as pd
import pytest


def make_series(start: str = "2014-01-01", days: int = 3 * 365, seed: int = 0) -> pd.Series:
    """Consommation fictive : niveau + saison annuelle + creux du week-end + bruit."""
    index = pd.date_range(start, periods=days, freq="D")
    rng = np.random.default_rng(seed)
    annee = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
    week_end = np.isin(index.dayofweek, [5, 6]).astype(float)
    valeurs = 1350 + 120 * np.cos(annee) - 200 * week_end + rng.normal(0, 15, days)
    return pd.Series(valeurs, index=index, name="consumption_gwh")


@pytest.fixture
def series() -> pd.Series:
    return make_series()
