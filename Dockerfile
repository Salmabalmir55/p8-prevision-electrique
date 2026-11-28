# Construction multi-étapes :
#   - api   : image de service légère (FastAPI + bibliothèques d'inférence), sans PyTorch ni DVC.
#   - train : image du pipeline (DVC, MLflow, PyTorch CPU) qui exécute `dvc repro`.
# Les dépendances sont installées dans un environnement virtuel, puis seul ce dernier
# est copié dans l'image finale.

ARG PYTHON_VERSION=3.11

# ---------- Socle commun ----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

# ---------- Dépendances ----------
# Le service n'installe que requirements/api.txt ; le pipeline y ajoute requirements/train.txt.
FROM base AS api-deps
RUN python -m venv /opt/venv
COPY requirements/ requirements/
RUN pip install -r requirements/api.txt

FROM api-deps AS train-deps
RUN pip install -r requirements/train.txt

# ---------- Socle d'exécution : utilisateur non privilégié ----------
FROM base AS runtime
# UID/GID fixes et explicites : identiques d'une construction à l'autre.
RUN groupadd --gid 999 app && useradd --uid 999 --gid 999 --home-dir /app --no-create-home app

# ---------- Image de service ----------
FROM runtime AS api
COPY --from=api-deps /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
RUN mkdir -p models && chown app:app models
USER app
ENV APP_ENV=dev \
    MODEL_PATH=/app/models/forecaster.joblib
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Image du pipeline ----------
FROM runtime AS train
COPY --from=train-deps /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app .dvc/config .dvc/config
COPY --chown=app:app dvc.yaml dvc.lock params.yaml .dvcignore ./
# /mlflow : stockage du serveur MLflow quand cette image sert aussi de serveur de suivi.
RUN mkdir -p data/raw data/processed models reports /mlflow \
    && chown -R app:app /app /mlflow
USER app
# HOME accessible en écriture pour les caches de DVC et matplotlib ; USER pour DVC.
# L'image n'a pas l'exécutable git : on évite l'avertissement de MLflow qui le cherche.
ENV HOME=/tmp \
    USER=app \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_PYTHON_REFRESH=quiet
CMD ["dvc", "repro"]
