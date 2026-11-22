"""Backtest à origine glissante : la bonne façon d'évaluer une prévision temporelle.

Pour chaque origine dans la période de test, le modèle (appris une seule fois sur le train)
reçoit tout l'historique connu à cette date et prévoit les `horizon` jours suivants.
On compare ensuite ces prévisions aux valeurs réellement observées.
"""

from __future__ import annotations

import pandas as pd

from src.common import calculate_metrics
from src.forecasters import Forecaster


def rolling_origin(
    forecaster: Forecaster,
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    step: int,
) -> tuple[dict, pd.DataFrame]:
    """Retourne les métriques agrégées et le détail (origine, date, réel, prévu)."""
    if horizon < 1 or step < 1:
        raise ValueError("L'horizon et le pas doivent être positifs.")
    horizon = min(horizon, len(test))
    complet = pd.concat([train, test])
    complet.index.freq = "D"
    lignes = []
    for decalage in range(0, len(test) - horizon + 1, step):
        historique = complet.iloc[: len(train) + decalage]
        reel = test.iloc[decalage : decalage + horizon]
        prevu = forecaster.predict(historique, horizon)
        origine = historique.index[-1]
        for date, valeur_reelle, valeur_prevue in zip(reel.index, reel.to_numpy(), prevu, strict=False):
            lignes.append(
                {
                    "origin": origine,
                    "date": date,
                    "actual": float(valeur_reelle),
                    "forecast": float(valeur_prevue),
                }
            )
    detail = pd.DataFrame(lignes)
    metriques = calculate_metrics(detail["actual"], detail["forecast"])
    metriques["n_origins"] = int(detail["origin"].nunique())
    return metriques, detail
