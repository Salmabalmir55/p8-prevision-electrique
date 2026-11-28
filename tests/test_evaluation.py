"""Métriques, sélection des modèles et contrôle qualité."""

import pytest

from src.common import calculate_metrics, validate_mae
from src.evaluate import quality_gate, select_models

RESULTATS = {
    "naive": {"short": {"mae": 50, "mape": 4.0}, "long": {"mae": 60, "mape": 5.0}},
    "ridge": {"short": {"mae": 30, "mape": 2.5}, "long": {"mae": 70, "mape": 6.0}},
    "prophet": {"short": {"mae": 40, "mape": 3.0}, "long": {"mae": 40, "mape": 3.5}},
}
SEUILS = {"max_mape_short": 4.0, "max_mape_long": 7.0, "must_beat_baseline": True}


def test_metrics_are_computed_in_gwh_and_percent():
    resultat = calculate_metrics([100, 200], [110, 190])
    assert resultat == {"mae": 10.0, "rmse": 10.0, "mape": pytest.approx(7.5)}


def test_selection_is_done_separately_for_each_horizon():
    assert select_models(RESULTATS) == {"short": "ridge", "long": "prophet"}


def test_quality_gate_accepts_good_models():
    assert quality_gate(RESULTATS, {"short": "ridge", "long": "prophet"}, SEUILS) == []


def test_quality_gate_refuses_a_model_worse_than_the_baseline():
    echecs = quality_gate(RESULTATS, {"short": "ridge", "long": "ridge"}, SEUILS)
    assert any("ne bat pas la référence" in echec for echec in echecs)


def test_quality_gate_refuses_a_model_above_the_threshold():
    seuils = {**SEUILS, "max_mape_short": 2.0}
    echecs = quality_gate(RESULTATS, {"short": "ridge", "long": "prophet"}, seuils)
    assert any("MAPE" in echec for echec in echecs)


def test_validate_mae():
    validate_mae(10.0, 10.0)
    with pytest.raises(ValueError, match="MAE trop élevée"):
        validate_mae(10.1, 10.0)
