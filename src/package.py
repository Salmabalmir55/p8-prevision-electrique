"""Réentraîne les modèles retenus sur toute la série et produit l'artefact servi par l'API.

Le backtest a choisi les modèles sans regarder la période de test ; une fois ce choix
validé, on les réentraîne sur toutes les données pour que la prévision parte du
dernier jour observé. L'artefact embarque sa trace (exécution MLflow, empreinte des
données, commit) pour qu'un modèle en service soit toujours relié à son origine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import mlflow
import pandas as pd

from src.common import ROOT, UNIT, git_commit, load_params, md5_file, read_series
from src.forecasters import build_forecaster
from src.tracking import configure_mlflow, resume_run

MODEL_PATH = ROOT / "models" / "forecaster.joblib"
HISTORY_DAYS = 730  # historique embarqué : assez pour tous les modèles, artefact léger


def build_bundle(full: pd.Series, selection: dict, params: dict, metrics: dict, trace: dict) -> dict:
    modeles = {}
    for terme in ("short", "long"):
        nom = selection[terme]["model"]
        if nom not in modeles:
            modeles[nom] = build_forecaster(nom, params["models"].get(nom, {})).fit(full)
    maintenant = datetime.now(UTC)
    return {
        "model_version": maintenant.strftime("%Y.%m.%d-%H%M%S"),
        "created_at": maintenant.isoformat(),
        "unit": UNIT,
        "short_term_max_days": int(params["serving"]["short_term_max_days"]),
        "short_term": {
            "name": selection["short"]["model"],
            "forecaster": modeles[selection["short"]["model"]],
        },
        "long_term": {"name": selection["long"]["model"], "forecaster": modeles[selection["long"]["model"]]},
        "backtest": {terme: selection[terme] for terme in ("short", "long")},
        "history": full.iloc[-HISTORY_DAYS:],
        "trace": trace,
        "all_models": dict(metrics["models"].items()),
    }


def main() -> None:
    params = load_params()
    rapport = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    if rapport["status"] != "passed":
        raise SystemExit("Contrôle qualité non validé : aucun modèle n'est publié.")
    candidats = joblib.load(ROOT / "models" / "candidates.joblib")
    train_path = ROOT / "data" / "processed" / "train.csv"
    test_path = ROOT / "data" / "processed" / "test.csv"
    full = pd.concat([read_series(train_path), read_series(test_path)])
    full.index.freq = "D"

    configure_mlflow()
    with resume_run(candidats["parent_run_id"], "package") as run:
        trace = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "git_commit": git_commit(),
            "data_train_md5": md5_file(train_path),
            "data_test_md5": md5_file(test_path),
        }
        bundle = build_bundle(full, rapport["selected"], params, rapport, trace)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        mlflow.set_tag("model_version", bundle["model_version"])
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

    print(
        f"Modèle {bundle['model_version']} publié dans {MODEL_PATH} "
        f"(court terme : {bundle['short_term']['name']}, long terme : {bundle['long_term']['name']}, "
        f"données jusqu'au {full.index.max().date()})."
    )


if __name__ == "__main__":
    main()
