"""API de prévision de la consommation électrique journalière (GWh).

Le même code sert les trois environnements ; seul APP_ENV change, ce qui charge
configs/<env>.yaml (horizon par défaut et maximal, niveau de journalisation).

Routes :
- GET  /health   sonde de vie (le processus répond) ;
- GET  /ready    sonde de disponibilité (le modèle est chargé) ;
- GET  /model    modèles servis, performances de backtest et traçabilité ;
- POST /predict  prévision court terme (≤ seuil) ou long terme.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common import ROOT

ENVIRONNEMENTS = {"dev", "staging", "prod"}
logger = logging.getLogger("api")


# --------------------------------------------------------------------------- configuration


class EnvironmentConfig(BaseModel):
    environment: str
    forecast_horizon_default: int = Field(ge=1)
    forecast_horizon_max: int = Field(ge=1)
    log_level: str = "info"


def load_environment_config(environment: str | None = None) -> EnvironmentConfig:
    nom = (environment or os.getenv("APP_ENV", "dev")).lower()
    if nom not in ENVIRONNEMENTS:
        raise RuntimeError(f"APP_ENV doit valoir dev, staging ou prod (reçu : {nom}).")
    with (ROOT / "configs" / f"{nom}.yaml").open(encoding="utf-8") as flux:
        config = EnvironmentConfig(**yaml.safe_load(flux))
    if config.forecast_horizon_default > config.forecast_horizon_max:
        raise RuntimeError("L'horizon par défaut dépasse l'horizon maximal.")
    return config


# --------------------------------------------------------------------------- schémas


class PredictionRequest(BaseModel):
    horizon_days: int | None = Field(
        default=None, ge=1, description="Nombre de jours à prévoir ; défaut selon l'environnement."
    )


class ForecastPoint(BaseModel):
    date: Date
    consumption_gwh: float


class PredictionResponse(BaseModel):
    environment: str
    model_version: str
    horizon_type: str
    model: str
    unit: str
    forecast: list[ForecastPoint]


# --------------------------------------------------------------------------- service


class ForecastService:
    """Charge l'artefact du pipeline et calcule les prévisions."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.history: pd.Series = bundle["history"]
        self.last_date = self.history.index[-1]
        self.short_max = int(bundle["short_term_max_days"])

    @classmethod
    def from_path(cls, path: Path) -> ForecastService:
        if not path.exists():
            raise FileNotFoundError(f"Modèle absent ({path}) : lancez `dvc repro`.")
        return cls(joblib.load(path))

    def term(self, horizon: int) -> str:
        return "short" if horizon <= self.short_max else "long"

    def model_name(self, horizon: int) -> str:
        return self.bundle[f"{self.term(horizon)}_term"]["name"]

    @lru_cache(maxsize=512)
    def forecast(self, horizon: int) -> tuple[float, ...]:
        modele = self.bundle[f"{self.term(horizon)}_term"]["forecaster"]
        return tuple(float(v) for v in modele.predict(self.history, horizon))

    def dates(self, horizon: int) -> pd.DatetimeIndex:
        return pd.date_range(self.last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def create_app(config: EnvironmentConfig | None = None, model_path: Path | str | None = None) -> FastAPI:
    """Fabrique de l'application : facilite les tests de chaque environnement."""
    config = config or load_environment_config()
    chemin = Path(model_path or os.getenv("MODEL_PATH", ROOT / "models" / "forecaster.joblib"))
    logging.basicConfig(level=config.log_level.upper())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Chargement au démarrage ; si le modèle manque, l'API démarre quand même
        # (/health répond) et /ready signale l'indisponibilité.
        charger_modele()
        yield

    app = FastAPI(
        title="Prévision de consommation électrique",
        version="1.0.0",
        description=f"Environnement **{config.environment}** — prévisions journalières en GWh.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.service = None

    def charger_modele() -> ForecastService | None:
        if app.state.service is None:
            try:
                app.state.service = ForecastService.from_path(chemin)
            except FileNotFoundError as erreur:
                logger.warning(str(erreur))
                return None
            logger.info("Modèle %s chargé depuis %s", app.state.service.bundle["model_version"], chemin)
        return app.state.service

    def service_requis() -> ForecastService:
        service = charger_modele()
        if service is None:
            raise HTTPException(status_code=503, detail="Modèle indisponible : lancez `dvc repro`.")
        return service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "environment": config.environment}

    @app.get("/ready")
    def ready() -> dict:
        service = service_requis()
        return {
            "status": "ready",
            "environment": config.environment,
            "model_version": service.bundle["model_version"],
        }

    @app.get("/model")
    def model() -> dict:
        service = service_requis()
        bundle = service.bundle
        return {
            "environment": config.environment,
            "model_version": bundle["model_version"],
            "created_at": bundle["created_at"],
            "data_until": service.last_date.date().isoformat(),
            "short_term": {
                "model": bundle["short_term"]["name"],
                "max_horizon_days": service.short_max,
                "backtest": bundle["backtest"]["short"],
            },
            "long_term": {
                "model": bundle["long_term"]["name"],
                "max_horizon_days": config.forecast_horizon_max,
                "backtest": bundle["backtest"]["long"],
            },
            "trace": bundle["trace"],
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(requete: PredictionRequest | None = None) -> dict:
        horizon = (requete.horizon_days if requete else None) or config.forecast_horizon_default
        if horizon > config.forecast_horizon_max:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Horizon maximal en {config.environment} : "
                    f"{config.forecast_horizon_max} jours (demandé : {horizon})."
                ),
            )
        service = service_requis()
        valeurs = service.forecast(horizon)
        return {
            "environment": config.environment,
            "model_version": service.bundle["model_version"],
            "horizon_type": service.term(horizon),
            "model": service.model_name(horizon),
            "unit": "GWh",
            "forecast": [
                {"date": jour.date(), "consumption_gwh": round(valeur, 3)}
                for jour, valeur in zip(service.dates(horizon), valeurs, strict=False)
            ],
        }

    return app


app = create_app()
