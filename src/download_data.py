"""Télécharge la consommation électrique journalière et vérifie son empreinte SHA-256.

Le fichier brut est conservé tel quel : la vérification d'empreinte garantit que
deux exécutions du pipeline partent exactement des mêmes données.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.common import ROOT, load_params

RAW_PATH = ROOT / "data" / "raw" / "opsd_germany_daily.csv"


def verify_sha256(content: bytes, expected: str) -> None:
    """Refuse un fichier dont le contenu a changé depuis la version de référence."""
    obtenu = hashlib.sha256(content).hexdigest()
    if obtenu != expected:
        raise ValueError(
            f"Empreinte SHA-256 inattendue : {obtenu} (attendu {expected}). "
            "La source a changé : vérifiez-la puis mettez à jour params.yaml."
        )


def fetch(url: str, attempts: int = 3, timeout: int = 60) -> bytes:
    """Télécharge le fichier, avec quelques nouvelles tentatives en cas d'erreur réseau."""
    requete = Request(url, headers={"User-Agent": "projet-p8-mlops/1.0"})
    for tentative in range(1, attempts + 1):
        try:
            with urlopen(requete, timeout=timeout) as reponse:
                return reponse.read()
        except (URLError, TimeoutError) as erreur:
            if tentative == attempts:
                raise
            print(f"Téléchargement échoué ({erreur}), nouvel essai {tentative + 1}/{attempts}.")
            time.sleep(2 * tentative)
    raise RuntimeError("inaccessible")  # pour l'analyse statique


def main(destination: Path = RAW_PATH) -> None:
    params = load_params()["data"]
    contenu = fetch(params["url"])
    verify_sha256(contenu, params["sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contenu)
    print(f"Données brutes enregistrées : {destination} ({len(contenu)} octets, empreinte vérifiée).")


if __name__ == "__main__":
    main()
