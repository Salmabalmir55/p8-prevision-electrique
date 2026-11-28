"""Backtest des candidats, sélection court/long terme et contrôle qualité avant mise en service.

Sorties :
- reports/metrics.json      métriques de chaque modèle et modèles retenus (suivi par DVC) ;
- reports/forecast_long.csv prévision longue de chaque modèle face au réel (graphique DVC) ;
- reports/figures/forecast_long.png même comparaison, en image pour le rapport.
L'étape échoue (code de sortie non nul) si le contrôle qualité est refusé : DVC
n'exécute alors pas l'étape package et aucun modèle dégradé n'est publié.
"""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib
import mlflow
import pandas as pd

from src.backtest import rolling_origin
from src.common import ROOT, load_params, md5_file, read_series
from src.tracking import configure_mlflow, resume_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = ROOT / "reports"
TERMES = ("short", "long")


def select_models(resultats: dict[str, dict]) -> dict[str, str]:
    """Pour chaque terme, retient le modèle de plus faible MAE en backtest."""
    return {terme: min(resultats, key=lambda nom: resultats[nom][terme]["mae"]) for terme in TERMES}


def quality_gate(resultats: dict, selection: dict, evaluation: dict) -> list[str]:
    """Liste des règles non respectées (vide si le modèle peut être mis en service)."""
    echecs = []
    for terme in TERMES:
        nom = selection[terme]
        mape = resultats[nom][terme]["mape"]
        seuil = float(evaluation[f"max_mape_{terme}"])
        if mape > seuil:
            echecs.append(f"{terme} : MAPE {mape:.2f} % > seuil {seuil:.2f} % ({nom})")
        if evaluation.get("must_beat_baseline") and "naive" in resultats and nom != "naive":
            reference = resultats["naive"][terme]["mae"]
            if resultats[nom][terme]["mae"] >= reference:
                echecs.append(f"{terme} : {nom} ne bat pas la référence naïve")
    return echecs


def plot_long_forecasts(detail: pd.DataFrame, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axe = plt.subplots(figsize=(11, 4.5))
    axe.plot(detail["date"], detail["actual"], color="black", linewidth=2, label="réel")
    for colonne in [c for c in detail.columns if c not in ("date", "actual")]:
        axe.plot(detail["date"], detail[colonne], linewidth=1.2, label=colonne)
    axe.set_title("Prévision à 180 jours depuis le 31/12/2016")
    axe.set_ylabel("Consommation (GWh / jour)")
    axe.legend(ncol=6, fontsize=8, loc="lower left")
    axe.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main() -> None:
    params = load_params()
    paquet = joblib.load(ROOT / "models" / "candidates.joblib")
    test_path = ROOT / "data" / "processed" / "test.csv"
    train = read_series(ROOT / "data" / "processed" / "train.csv")
    test = read_series(test_path)
    configure_mlflow()

    resultats: dict[str, dict] = {}
    comparaison = pd.DataFrame({"date": test.index, "actual": test.to_numpy()})
    with resume_run(paquet["parent_run_id"], "evaluate"):
        for nom, candidat in paquet["candidates"].items():
            modele = candidat["forecaster"]
            resultats[nom] = {}
            with resume_run(candidat["run_id"], nom, nested=True):
                for terme in TERMES:
                    reglage = params["backtest"][terme]
                    metriques, detail = rolling_origin(
                        modele, train, test, reglage["horizon_days"], reglage["step_days"]
                    )
                    resultats[nom][terme] = metriques
                    mlflow.log_metrics({f"{terme}_{cle}": valeur for cle, valeur in metriques.items()})
                    if terme == "long":
                        premiere = detail[detail["origin"] == detail["origin"].min()]
                        comparaison = comparaison.merge(
                            premiere[["date", "forecast"]].rename(columns={"forecast": nom}),
                            on="date",
                            how="left",
                        )
            print(
                f"{nom:8s} court terme MAE {resultats[nom]['short']['mae']:7.2f} GWh"
                f" | long terme MAE {resultats[nom]['long']['mae']:7.2f} GWh"
            )

        selection = select_models(resultats)
        echecs = quality_gate(resultats, selection, params["evaluation"])
        rapport = {
            "unit": "GWh",
            "selected": {terme: {"model": nom, **resultats[nom][terme]} for terme, nom in selection.items()},
            "models": resultats,
            "thresholds": params["evaluation"],
            "data_test_md5": md5_file(test_path),
            "status": "failed" if echecs else "passed",
            "failures": echecs,
        }
        REPORTS.mkdir(parents=True, exist_ok=True)
        # Fins de ligne LF : mêmes empreintes DVC sous Windows et sous Linux.
        (REPORTS / "metrics.json").write_text(json.dumps(rapport, indent=2), encoding="utf-8", newline="\n")
        comparaison = comparaison.dropna()
        comparaison.to_csv(
            REPORTS / "forecast_long.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n"
        )
        plot_long_forecasts(comparaison, REPORTS / "figures" / "forecast_long.png")

        mlflow.set_tags(
            {
                "selected_short": selection["short"],
                "selected_long": selection["long"],
                "quality_gate": rapport["status"],
            }
        )
        for terme, nom in selection.items():
            mlflow.log_metric(f"selected_{terme}_mae", resultats[nom][terme]["mae"])
            mlflow.log_metric(f"selected_{terme}_mape", resultats[nom][terme]["mape"])
        mlflow.log_artifact(str(REPORTS / "metrics.json"))
        mlflow.log_artifact(str(REPORTS / "figures" / "forecast_long.png"))

    print(f"Modèles retenus : court terme = {selection['short']}, long terme = {selection['long']}")
    if echecs:
        print("Contrôle qualité REFUSÉ :\n- " + "\n- ".join(echecs))
        sys.exit(1)
    print("Contrôle qualité validé.")


if __name__ == "__main__":
    main()
