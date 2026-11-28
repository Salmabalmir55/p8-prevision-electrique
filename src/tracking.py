"""Configuration du suivi d'expériences MLflow, partagée par les étapes du pipeline."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import mlflow

from src.common import ROOT

EXPERIMENT = "prevision-consommation-electrique"


def configure_mlflow() -> str:
    """Utilise MLFLOW_TRACKING_URI si le serveur répond, sinon le dossier local ./mlruns."""
    distant = os.getenv("MLFLOW_TRACKING_URI")
    local = (ROOT / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(distant or local)
    try:
        mlflow.set_experiment(EXPERIMENT)
    except Exception as erreur:  # serveur absent ou injoignable
        print(f"MLflow distant indisponible ({erreur}) ; suivi local dans ./mlruns.")
        mlflow.set_tracking_uri(local)
        mlflow.set_experiment(EXPERIMENT)
    return mlflow.get_tracking_uri()


@contextmanager
def resume_run(run_id: str | None, run_name: str, nested: bool = False):
    """Rouvre une exécution créée par une étape précédente, ou en crée une nouvelle.

    Une nouvelle exécution est créée si l'étape précédente a écrit dans un autre
    serveur MLflow (par exemple train en local puis evaluate dans Docker).
    """
    try:
        run = mlflow.start_run(run_id=run_id, nested=nested) if run_id else None
    except Exception:
        run = None
    if run is None:
        run = mlflow.start_run(run_name=run_name, nested=nested)
        mlflow.set_tag("resumed_from", run_id or "aucune")
    try:
        yield run
    finally:
        mlflow.end_run()


def flatten(values: dict, prefix: str = "") -> dict:
    """Aplatit un dictionnaire imbriqué en clés « a.b.c » pour mlflow.log_params."""
    resultat = {}
    for cle, valeur in values.items():
        nom = f"{prefix}.{cle}" if prefix else str(cle)
        if isinstance(valeur, dict):
            resultat.update(flatten(valeur, nom))
        elif isinstance(valeur, list | tuple):
            resultat[nom] = json.dumps(list(valeur))
        else:
            resultat[nom] = valeur
    return resultat
