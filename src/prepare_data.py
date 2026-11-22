"""Nettoie la série de consommation, contrôle sa qualité et la découpe dans le temps."""

from __future__ import annotations

import json

import pandas as pd

from src.common import ROOT, TARGET, load_params

PROCESSED = ROOT / "data" / "processed"


def clean_series(raw: pd.DataFrame, min_value: float, max_value: float) -> tuple[pd.DataFrame, dict]:
    """Retourne une série quotidienne continue (date, consumption_gwh) et un rapport qualité.

    - dates invalides ou dupliquées retirées (on garde la dernière observation) ;
    - valeurs hors bornes physiques considérées comme manquantes ;
    - jours manquants interpolés linéairement.
    """
    manquantes = {"Date", "Consumption"} - set(raw.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier brut : {sorted(manquantes)}")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            TARGET: pd.to_numeric(raw["Consumption"], errors="coerce"),
        }
    )
    lignes_brutes = len(frame)
    frame = frame.dropna(subset=["date"]).sort_values("date")
    doublons = int(frame.duplicated("date").sum())
    frame = frame.drop_duplicates("date", keep="last").set_index("date").asfreq("D")
    frame.index.name = "date"

    hors_bornes = (frame[TARGET] < min_value) | (frame[TARGET] > max_value)
    nb_hors_bornes = int(hors_bornes.sum())
    frame.loc[hors_bornes, TARGET] = float("nan")
    nb_manquantes = int(frame[TARGET].isna().sum())
    frame[TARGET] = frame[TARGET].interpolate(method="time").ffill().bfill()

    if frame[TARGET].isna().any() or frame.empty:
        raise ValueError("Aucune valeur de consommation exploitable.")

    rapport = {
        "raw_rows": lignes_brutes,
        "days": len(frame),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "duplicates_removed": doublons,
        "out_of_range_values": nb_hors_bornes,
        "interpolated_days": nb_manquantes,
        "min_gwh": round(float(frame[TARGET].min()), 3),
        "max_gwh": round(float(frame[TARGET].max()), 3),
        "mean_gwh": round(float(frame[TARGET].mean()), 3),
    }
    return frame.reset_index(), rapport


def split_by_date(frame: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage chronologique : jamais de mélange aléatoire sur une série temporelle."""
    debut = pd.Timestamp(test_start)
    train = frame[frame["date"] < debut]
    test = frame[frame["date"] >= debut]
    if train.empty or test.empty:
        raise ValueError(f"La date {test_start} ne laisse pas de données des deux côtés.")
    return train.copy(), test.copy()


def main() -> None:
    params = load_params()
    raw = pd.read_csv(ROOT / "data" / "raw" / "opsd_germany_daily.csv")
    frame, rapport = clean_series(raw, params["data"]["min_gwh"], params["data"]["max_gwh"])
    train, test = split_by_date(frame, params["split"]["test_start"])
    rapport.update({"train_days": len(train), "test_days": len(test)})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    # Fins de ligne LF imposées : sinon Windows écrit des CRLF et les empreintes DVC
    # diffèrent d'un système à l'autre pour des données identiques.
    csv = {"index": False, "date_format": "%Y-%m-%d", "lineterminator": "\n"}
    train.to_csv(PROCESSED / "train.csv", **csv)
    test.to_csv(PROCESSED / "test.csv", **csv)
    destination = ROOT / "reports" / "data_quality.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rapport, indent=2), encoding="utf-8", newline="\n")
    print(f"Préparation terminée : {len(train)} jours train, {len(test)} jours test.")


if __name__ == "__main__":
    main()
