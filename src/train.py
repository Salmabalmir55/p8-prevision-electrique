"""Entraîne chaque modèle candidat sur la période d'apprentissage et le trace dans MLflow.

Une exécution MLflow parente regroupe le passage complet du pipeline ; chaque modèle
y a son exécution enfant (hyperparamètres, durée d'apprentissage). Les étapes
evaluate et package rouvrent ces mêmes exécutions pour y ajouter leurs résultats.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import joblib
import mlflow

from src.common import ROOT, git_commit, load_params, md5_file, read_series
from src.forecasters import build_forecaster
from src.tracking import configure_mlflow, flatten

CANDIDATES_PATH = ROOT / "models" / "candidates.joblib"


def main() -> None:
    params = load_params()
    train_path = ROOT / "data" / "processed" / "train.csv"
    train = read_series(train_path)
    tracking_uri = configure_mlflow()
    horodatage = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    candidats = {}
    with mlflow.start_run(run_name=f"pipeline-{horodatage}") as parent:
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_train_md5": md5_file(train_path),
                "train_start": train.index.min().date().isoformat(),
                "train_end": train.index.max().date().isoformat(),
            }
        )
        mlflow.log_params(flatten({"split": params["split"], "candidates": params["candidates"]}))
        mlflow.log_artifact(str(ROOT / "params.yaml"))

        for nom in params["candidates"]:
            with mlflow.start_run(run_name=nom, nested=True) as enfant:
                modele = build_forecaster(nom, params["models"].get(nom, {}))
                debut = time.perf_counter()
                modele.fit(train)
                duree = time.perf_counter() - debut
                mlflow.set_tag("model", nom)
                mlflow.log_params(flatten(modele.get_params()))
                mlflow.log_metric("fit_seconds", duree)
                if hasattr(modele, "train_loss_"):
                    mlflow.log_metric("train_loss", modele.train_loss_)
                candidats[nom] = {"forecaster": modele, "run_id": enfant.info.run_id}
                print(f"{nom:8s} entraîné en {duree:6.1f} s")

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "parent_run_id": parent.info.run_id,
            "tracking_uri": tracking_uri,
            "candidates": candidats,
        },
        CANDIDATES_PATH,
    )
    print(f"Exécution MLflow : {parent.info.run_id} ({tracking_uri})")


if __name__ == "__main__":
    main()
