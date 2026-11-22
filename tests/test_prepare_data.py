"""Nettoyage, contrôle qualité et découpage chronologique."""

import hashlib

import pandas as pd
import pytest

from src.download_data import verify_sha256
from src.prepare_data import clean_series, split_by_date


def raw_frame():
    return pd.DataFrame(
        {
            "Date": [
                "2016-01-01",
                "2016-01-02",
                "2016-01-02",
                "2016-01-04",
                "pas-une-date",
                "2016-01-05",
            ],
            "Consumption": [1000, 1100, 1200, 1300, 999, 50000],
            "Wind": [None] * 6,
        }
    )


def test_clean_series_fills_gaps_and_reports_quality():
    frame, rapport = clean_series(raw_frame(), 500, 2000)
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2016-01-01",
        "2016-01-02",
        "2016-01-03",
        "2016-01-04",
        "2016-01-05",
    ]
    # Doublon : on garde la dernière valeur ; trou du 03 interpolé ; 50000 hors bornes.
    assert frame["consumption_gwh"].tolist()[:4] == [1000, 1200, 1250, 1300]
    assert frame["consumption_gwh"].iloc[-1] == 1300
    assert rapport["duplicates_removed"] == 1
    assert rapport["out_of_range_values"] == 1
    assert rapport["interpolated_days"] == 2
    assert not frame["consumption_gwh"].isna().any()


def test_clean_series_rejects_missing_columns():
    with pytest.raises(ValueError, match="Colonnes absentes"):
        clean_series(pd.DataFrame({"Date": ["2016-01-01"]}), 500, 2000)


def test_split_by_date_never_mixes_past_and_future():
    frame, _ = clean_series(raw_frame(), 500, 2000)
    train, test = split_by_date(frame, "2016-01-04")
    assert train["date"].max() < test["date"].min()
    assert len(train) == 3
    assert len(test) == 2


def test_split_by_date_requires_data_on_both_sides():
    frame, _ = clean_series(raw_frame(), 500, 2000)
    with pytest.raises(ValueError):
        split_by_date(frame, "2030-01-01")


def test_sha256_verification_detects_changed_source():
    contenu = b"Date,Consumption\n"
    empreinte = hashlib.sha256(contenu).hexdigest()
    verify_sha256(contenu, empreinte)
    with pytest.raises(ValueError, match="Empreinte"):
        verify_sha256(contenu + b"x", empreinte)
