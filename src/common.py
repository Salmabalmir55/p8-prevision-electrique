"""Fonctions partagées : chemins, paramètres, métriques, traçabilité."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET = "consumption_gwh"
UNIT = "GWh"


def load_params(path: Path | str | None = None) -> dict:
    """Charge le fichier de paramètres centralisé (params.yaml)."""
    fichier = Path(path) if path else ROOT / "params.yaml"
    with fichier.open(encoding="utf-8") as entree:
        return yaml.safe_load(entree)


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE (en GWh) et MAPE (en %) sous forme de nombres Python."""
    vrai = np.asarray(y_true, dtype=float)
    prevu = np.asarray(y_pred, dtype=float)
    if vrai.shape != prevu.shape or vrai.size == 0:
        raise ValueError("Les séries réelle et prévue doivent avoir la même taille non nulle.")
    erreur = prevu - vrai
    return {
        "mae": float(np.mean(np.abs(erreur))),
        "rmse": float(np.sqrt(np.mean(erreur**2))),
        "mape": float(np.mean(np.abs(erreur) / np.abs(vrai)) * 100),
    }


def validate_mae(mae: float, max_mae: float, label: str = "") -> None:
    """Refuse un modèle dont l'erreur absolue moyenne dépasse le seuil."""
    if mae > max_mae:
        prefixe = f"{label} : " if label else ""
        raise ValueError(
            f"{prefixe}MAE trop élevée : {mae:.2f} {UNIT}, limite autorisée : {max_mae:.2f} {UNIT}."
        )


def read_series(path: Path | str) -> pd.Series:
    """Lit un CSV (date, consumption_gwh) en série quotidienne indexée par date."""
    frame = pd.read_csv(path, parse_dates=["date"])
    series = frame.set_index("date")[TARGET].astype(float)
    series.index.freq = "D"
    return series


def md5_file(path: Path | str) -> str:
    """Empreinte d'un fichier, journalisée dans MLflow pour relier une exécution à ses données."""
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for bloc in iter(lambda: stream.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()


def read_git_head(root: Path) -> str | None:
    """Lit le commit courant directement dans .git, sans exécutable git.

    Utile dans les conteneurs : l'image n'a pas git, mais le dépôt y est monté.
    """
    git = root / ".git"
    try:
        tete = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not tete.startswith("ref: "):
            return tete  # HEAD détachée : le commit est écrit directement
        reference = tete[5:]
        fichier = git / reference
        if fichier.exists():
            return fichier.read_text(encoding="utf-8").strip()
        for ligne in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if ligne.endswith(" " + reference):
                return ligne.split(" ")[0]
    except OSError:
        return None
    return None


def git_commit() -> str:
    """Commit Git courant, ou « inconnu » hors dépôt Git."""
    try:
        resultat = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return resultat.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return read_git_head(ROOT) or "inconnu"
